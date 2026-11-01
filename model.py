"""Tiny GPT-2-style protein LM in JAX (stand-in for ProtGPT2) with:
  * KV-cached batched generation
  * L2 constrained decoding (vocabulary mask)
  * L3 running linear probe on pooled hidden states (blocks the request mid-generation)
  * activation steering at a fixed layer (arm D)
  * LoRA adapters on all four linear maps of every block (arm E, representation rerouting)
"""
import functools, math, os, json, time
import numpy as np
import jax, jax.numpy as jnp
import optax
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from common import AA_SET, EOS

OUT = os.environ.get("PG_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(OUT, exist_ok=True)

D, H, NL, NPOS = 128, 4, 4, 320
DH = D // H
STEER_LAYER = 2          # steering applied after block 2 (hs index 2)
MAX_PROMPT_TOK = 200
MAX_NEW = 72


# ----------------------------------------------------------------------------- tokenizer
def train_tokenizer(seqs, vocab=1024):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(seqs, trainer)
    return tok


def load_tok():
    return Tokenizer.from_file(os.path.join(OUT, "tok.json"))


def residue_token_mask(tok):
    V = tok.get_vocab_size()
    ok = np.zeros(V, bool)
    for i in range(V):
        s = tok.decode([i])
        if s and all(c in AA_SET for c in s):
            ok[i] = True
    ok[tok.token_to_id(EOS)] = True
    return jnp.asarray(ok)


def pad_batch(tok, seqs, max_len=300, left=False):
    eos = tok.token_to_id(EOS)
    enc = [([eos] + tok.encode(s).ids + [eos])[:max_len] for s in seqs]
    L = max(len(e) for e in enc)
    X = np.full((len(enc), L), eos, np.int32); M = np.zeros((len(enc), L), bool)
    for i, e in enumerate(enc):
        X[i, :len(e)] = e; M[i, :len(e)] = True
    return X, M


# ----------------------------------------------------------------------------- model
def init_params(key, V):
    ks = jax.random.split(key, 2 + NL)
    nrm = lambda k, s, std=0.02: jax.random.normal(k, s) * std
    blocks = []
    for l in range(NL):
        k = jax.random.split(ks[2 + l], 4)
        blocks.append(dict(
            ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D), qkv_w=nrm(k[0], (D, 3 * D)), qkv_b=jnp.zeros(3 * D),
            proj_w=nrm(k[1], (D, D), 0.02 / math.sqrt(2 * NL)), proj_b=jnp.zeros(D),
            ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D), fc_w=nrm(k[2], (D, 4 * D)), fc_b=jnp.zeros(4 * D),
            out_w=nrm(k[3], (4 * D, D), 0.02 / math.sqrt(2 * NL)), out_b=jnp.zeros(D)))
    return dict(wte=nrm(ks[0], (V, D)), wpe=nrm(ks[1], (NPOS, D), 0.01), blocks=blocks,
                lnf_g=jnp.ones(D), lnf_b=jnp.zeros(D))


def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g + b


LORA_NAMES = ("qkv_w", "proj_w", "fc_w", "out_w")


def eff(b, lora_l, name, scale):
    w = b[name]
    if lora_l is not None:
        A, B = lora_l[name]
        w = w + scale * (A @ B)
    return w


def mlp(x, b, lora_l, scale):
    h = ln(x, b["ln2_g"], b["ln2_b"])
    h = jax.nn.gelu(h @ eff(b, lora_l, "fc_w", scale) + b["fc_b"])
    return x + h @ eff(b, lora_l, "out_w", scale) + b["out_b"]


def forward(P, lora, ids, mask, pos, steer_u, steer_alpha, want_kv=False, lora_scale=2.0):
    """Full-sequence forward. ids/mask/pos: (B,T). Returns logits (B,T,V), hs list (NL+1 x (B,T,D))."""
    B, T = ids.shape
    x = P["wte"][ids] + P["wpe"][pos]
    causal = jnp.tril(jnp.ones((T, T), bool))
    allow = causal[None, :, :] & mask[:, None, :]
    bias = jnp.where(allow, 0.0, -1e9)[:, None]
    hs = [x]; kvs = []
    for l, b in enumerate(P["blocks"]):
        ll = None if lora is None else lora[l]
        h = ln(x, b["ln1_g"], b["ln1_b"])
        qkv = h @ eff(b, ll, "qkv_w", lora_scale) + b["qkv_b"]
        q, k, v = jnp.split(qkv, 3, -1)
        sp = lambda t: t.reshape(B, T, H, DH).transpose(0, 2, 1, 3)
        q, k, v = sp(q), sp(k), sp(v)
        att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(DH) + bias, -1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        x = x + o @ eff(b, ll, "proj_w", lora_scale) + b["proj_b"]
        x = mlp(x, b, ll, lora_scale)
        if l + 1 == STEER_LAYER:
            x = x - steer_alpha * steer_u
        hs.append(x)
        kvs.append((k, v))
    logits = ln(x, P["lnf_g"], P["lnf_b"]) @ P["wte"].T
    return (logits, hs, kvs) if want_kv else (logits, hs)


@functools.partial(jax.jit, static_argnames=("total", "pl"))
def prefill(P, lora, ids, mask, pos, steer_u, steer_alpha, total, pl):
    logits, hs, kvs = forward(P, lora, ids, mask, pos, steer_u, steer_alpha, want_kv=True)
    pad = total - ids.shape[1]
    kc = [jnp.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0))) for k, _ in kvs]
    vc = [jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0))) for _, v in kvs]
    return logits[:, -1], kc, vc, hs[pl]


@functools.partial(jax.jit, static_argnames=("pl",))
def step(P, lora, steer_u, steer_alpha, kc, vc, tok, pos, idx, keymask, pl, lora_scale=2.0):
    B = tok.shape[0]
    x = (P["wte"][tok] + P["wpe"][pos])[:, None, :]
    bias = jnp.where(keymask, 0.0, -1e9)[:, None, None, :]
    hp = None
    nk, nv = [], []
    for l, b in enumerate(P["blocks"]):
        ll = None if lora is None else lora[l]
        h = ln(x, b["ln1_g"], b["ln1_b"])
        qkv = h @ eff(b, ll, "qkv_w", lora_scale) + b["qkv_b"]
        q, k, v = jnp.split(qkv, 3, -1)
        sp = lambda t: t.reshape(B, 1, H, DH).transpose(0, 2, 1, 3)
        q, k, v = sp(q), sp(k), sp(v)
        K = jax.lax.dynamic_update_slice(kc[l], k, (0, 0, idx, 0))
        V = jax.lax.dynamic_update_slice(vc[l], v, (0, 0, idx, 0))
        att = jax.nn.softmax(q @ K.transpose(0, 1, 3, 2) / math.sqrt(DH) + bias, -1)
        o = (att @ V).transpose(0, 2, 1, 3).reshape(B, 1, D)
        x = x + o @ eff(b, ll, "proj_w", lora_scale) + b["proj_b"]
        x = mlp(x, b, ll, lora_scale)
        if l + 1 == STEER_LAYER:
            x = x - steer_alpha * steer_u
        nk.append(K); nv.append(V)
        if l + 1 == pl:
            hp = x[:, 0]
    logits = (ln(x, P["lnf_g"], P["lnf_b"]) @ P["wte"].T)[:, 0]
    return logits, nk, nv, hp


@functools.partial(jax.jit, static_argnames=("top_k",))
def sample(key, logits, vmask, top_k=40):
    logits = jnp.where(vmask[None, :], logits, -1e9)
    kth = jax.lax.top_k(logits, top_k)[0][:, -1:]
    logits = jnp.where(logits < kth, -1e9, logits)
    return jax.random.categorical(key, logits, axis=-1)


# ----------------------------------------------------------------------------- probe object
class Probe:
    def __init__(self, w, b, mu, sd, layer, tau=0.5):
        self.w, self.b, self.mu, self.sd, self.layer, self.tau = (jnp.asarray(w), float(b), jnp.asarray(mu),
                                                                jnp.asarray(sd), layer, tau)

    def prob(self, pooled):
        return jax.nn.sigmoid(((pooled - self.mu) / self.sd) @ self.w + self.b)


class Steer:
    def __init__(self, u, alpha):
        self.u, self.alpha = jnp.asarray(u), float(alpha)


def _bucket(n, sizes):
    for s in sizes:
        if n <= s:
            return s
    return sizes[-1]


def generate(P, tok, prompts, key, vocab_mask=None, probe=None, steer=None, lora=None,
             probe_every=8, max_new=MAX_NEW, top_k=40):
    """Returns list of dict(text, blocked, max_p, step). Prompts must already satisfy the context limit."""
    n = len(prompts)
    nb = _bucket(n, [64, 128, 256, 512, 1024])
    eos = tok.token_to_id(EOS)
    seqs = [[eos] + tok.encode(p).ids for p in prompts] + [[eos]] * (nb - n)
    L = _bucket(max(len(s) for s in seqs), [16, 32, 48, 64, 96, 128, 160, 201])
    ids = np.full((nb, L), eos, np.int32); mask = np.zeros((nb, L), bool)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = s; mask[i, L - len(s):] = True
    pos = np.clip(np.cumsum(mask, 1) - 1, 0, None).astype(np.int32)
    su = steer.u if steer else jnp.zeros(D); sa = steer.alpha if steer else 0.0
    pl = probe.layer if probe else 3
    total = L + max_new
    logits, kc, vc, hs = prefill(P, lora, jnp.asarray(ids), jnp.asarray(mask), jnp.asarray(pos), su, sa,
                                 total=total, pl=pl)
    m = mask.copy().astype(np.float32)
    for i, s in enumerate(seqs):
        m[i, L - len(s)] = 0.0                         # drop BOS from the pool
    pooled = (hs * jnp.asarray(m)[:, :, None]).sum(1); cnt = jnp.asarray(m.sum(1))
    max_p = np.zeros(nb); blocked = np.zeros(nb, bool); bstep = np.full(nb, -1)
    finished = np.zeros(nb, bool); gen = [[] for _ in range(nb)]
    vm = vocab_mask if vocab_mask is not None else jnp.ones(P["wte"].shape[0], bool)

    def check(t):
        if probe is None:
            return
        p = np.asarray(probe.prob(pooled / jnp.maximum(cnt, 1)[:, None]))
        newly = (~blocked) & (p > probe.tau)
        max_p[:] = np.maximum(max_p, p); bstep[newly] = t; blocked[:] = blocked | newly

    check(0)
    cur_pos = pos[:, -1].copy()
    keymask = np.zeros((nb, total), bool); keymask[:, :L] = mask
    for t in range(1, max_new + 1):
        key, sk = jax.random.split(key)
        nxt = np.asarray(sample(sk, logits, vm, top_k=top_k))
        dead = finished | blocked
        for i in np.nonzero(~dead)[0]:
            if nxt[i] == eos:
                finished[i] = True
            else:
                gen[i].append(int(nxt[i]))
        if (finished | blocked).all():
            break
        nxt = np.where(finished | blocked, eos, nxt).astype(np.int32)
        cur_pos = cur_pos + 1
        keymask[:, L + t - 1] = True
        logits, kc, vc, hp = step(P, lora, su, sa, kc, vc, jnp.asarray(nxt), jnp.asarray(cur_pos.astype(np.int32)),
                                  jnp.int32(L + t - 1), jnp.asarray(keymask), pl=pl)
        alive = jnp.asarray((~(finished | blocked)).astype(np.float32))
        pooled = pooled + hp * alive[:, None]; cnt = cnt + alive
        if probe is not None and t % probe_every == 0:
            check(t)
    check(max_new)
    return [dict(text=("" if blocked[i] else tok.decode(gen[i])), blocked=bool(blocked[i]),
                 max_p=float(max_p[i]), step=int(bstep[i])) for i in range(n)]


# ----------------------------------------------------------------------------- training / eval utils
def lm_loss(P, ids, mask, pos):
    logits, _ = forward(P, None, ids, mask, pos, jnp.zeros(D), 0.0)
    lp = jax.nn.log_softmax(logits[:, :-1], -1)
    tgt = ids[:, 1:]
    nll = -jnp.take_along_axis(lp, tgt[..., None], -1)[..., 0]
    m = mask[:, 1:].astype(jnp.float32)
    return (nll * m).sum() / m.sum()


def perplexity(P, tok, seqs, lora=None, steer=None, bs=100):
    su = steer.u if steer else jnp.zeros(D); sa = steer.alpha if steer else 0.0
    tot, n = 0.0, 0
    f = jax.jit(lambda P, lora, ids, mask, pos, su, sa: _nll(P, lora, ids, mask, pos, su, sa))
    for i in range(0, len(seqs), bs):
        X, M = pad_batch(tok, seqs[i:i + bs])
        pos = np.clip(np.cumsum(M, 1) - 1, 0, None).astype(np.int32)
        s, c = f(P, lora, jnp.asarray(X), jnp.asarray(M), jnp.asarray(pos), su, sa)
        tot += float(s); n += float(c)
    return math.exp(tot / n)


def _nll(P, lora, ids, mask, pos, su, sa):
    logits, _ = forward(P, lora, ids, mask, pos, su, sa)
    lp = jax.nn.log_softmax(logits[:, :-1], -1)
    nll = -jnp.take_along_axis(lp, ids[:, 1:][..., None], -1)[..., 0]
    m = mask[:, 1:].astype(jnp.float32)
    return (nll * m).sum(), m.sum()


def train_lm(tok, groups, steps=4000, bs=32, lr=3e-3, seed=0, log=print):
    """groups: list of (sequences, sampling_probability)."""
    key = jax.random.PRNGKey(seed)
    P = init_params(key, tok.get_vocab_size())
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, 150, steps, lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    st = opt.init(P)

    @jax.jit
    def upd(P, st, ids, mask, pos):
        loss, g = jax.value_and_grad(lm_loss)(P, ids, mask, pos)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, loss

    rng = np.random.default_rng(seed); t0 = time.time()
    for s in range(steps):
        gi = rng.choice(len(groups), size=bs, p=[g[1] for g in groups])
        batch = [groups[g][0][rng.integers(len(groups[g][0]))] for g in gi]
        X, M = pad_batch(tok, batch, 100)
        # fix shape to avoid recompiles
        Lf = 100
        Xp = np.full((bs, Lf), X[0, 0], np.int32); Mp = np.zeros((bs, Lf), bool)
        Xp[:, :X.shape[1]] = X; Mp[:, :X.shape[1]] = M
        pos = np.clip(np.cumsum(Mp, 1) - 1, 0, None).astype(np.int32)
        P, st, loss = upd(P, st, jnp.asarray(Xp), jnp.asarray(Mp), jnp.asarray(pos))
        if s % 250 == 0 or s == steps - 1:
            log(f"step {s} loss {float(loss):.3f} t={time.time()-t0:.0f}s")
    return P


def save_params(P, path):
    flat, tree = jax.tree_util.tree_flatten(P)
    np.savez(path, *[np.asarray(a) for a in flat])


def load_params(path, V):
    ref = init_params(jax.random.PRNGKey(0), V)
    flat, tree = jax.tree_util.tree_flatten(ref)
    z = np.load(path)
    return jax.tree_util.tree_unflatten(tree, [jnp.asarray(z[f"arr_{i}"]) for i in range(len(flat))])
