"""Step 1: build synthetic data, train tokenizer + base LM, calibrate the family oracle, sanity-check."""
import sys, json, os, time
import numpy as np
import jax, jax.numpy as jnp
import model as M
from data import build_dataset
from oracle import Oracle

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
sp = json.load(open(os.path.join(M.OUT, "splits.json")))      # real Swiss-Prot splits from data_real.py
S_lm = sp["S_train"] + sp["S_attacker"]; H_lm = sp["H_train"] + sp["H_attacker"]
groups = [(sp["B_train_bg"], 0.25), (sp["B_train_fam"], 0.35), (S_lm, 0.2), (H_lm, 0.2)]
tok = M.train_tokenizer(sp["B_train"] + S_lm + H_lm)
tok.save(os.path.join(M.OUT, "tok.json"))
print("vocab", tok.get_vocab_size(), "residues/token", np.mean([len(s) / len(tok.encode(s).ids) for s in sp["B_test"][:200]]))
P = M.train_lm(tok, groups, steps=steps)
M.save_params(P, os.path.join(M.OUT, "base.npz"))
ppl = {k: M.perplexity(P, tok, v[:200]) for k, v in [("B_test", sp["B_test"]), ("S_test", sp["S_test"]),
                                                    ("H_test", sp["H_test"])]}
print("perplexity", ppl)
json.dump(ppl, open(os.path.join(M.OUT, "base_ppl.json"), "w"))

# ---- oracle calibration (target S): references S_oracle; negatives = benign + H (+ random)
orS = Oracle(sp["S_oracle"]); T, tpr = orS.calibrate(sp["S_test"], sp["B_test"][:300] + sp["H_test"][:100])
orH = Oracle(sp["H_oracle"]); TH, tprH = orH.calibrate(sp["H_test"][:60], sp["B_test"][:300] + sp["S_test"])
print("oracle S: T=%.1f (neg99=%.1f, pos_med=%.1f) TPR=%.2f | oracle H: T=%.1f TPR=%.2f" % (T, orS.neg_q, orS.pos_med, tpr, TH, tprH))

# ---- sanity: seeded generation with the *undefended* model, plain S prefix seeds
vm = M.residue_token_mask(tok)
key = jax.random.PRNGKey(1)
for name, seqs, orc in [("S", sp["S_attacker"], orS), ("H", sp["H_attacker"], orH), ("B", sp["B_test_fam"][:30], orS)]:
    prompts = [s[:15] for s in seqs] * 3
    t0 = time.time()
    out = M.generate(P, tok, prompts, key)
    rate = np.mean([orc.is_member(o["text"]) for o in out])
    print(f"plain 15-aa seeds from {name}: continuation matches S-oracle(H-oracle for H) rate = {rate:.2f} "
          f"({len(prompts)} gens, {time.time()-t0:.1f}s); mean continuation residues "
          f"{np.mean([len(o['text']) for o in out]):.0f}")
json.dump(dict(T_S=T, tpr_S=tpr, T_H=TH, tpr_H=tprH), open(os.path.join(M.OUT, "oracle.json"), "w"))
