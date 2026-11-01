"""Synthetic protein families (stand-in for UniProtKB/Swiss-Prot partitions).

Layout (mirrors the proposal's partitions):
  screened family S  -> probe / rerouting training (S_train) + attacker-known + oracle refs + test
  held-out family H  -> never used for defense training (generalisation test, RQ2)
  benign families    -> utility set and negatives for defense training
S and H descend from a common super-ancestor (distant relatives), so H generalisation is a
non-trivial but plausible question. Benign families are unrelated.
"""
import numpy as np
from common import AA, BG_VEC


def _sample_seq(rng, L, comp):
    return "".join(rng.choice(list(AA), size=L, p=comp))


def _family_comp(rng, conc=300.0):
    comp = rng.dirichlet(BG_VEC * conc)
    return comp


def mutate(rng, anc, identity, comp, conserved=None):
    """Star-phylogeny member: substitutions (+ rare indels); conserved sites mutate 5x slower."""
    out = []
    L = len(anc)
    if conserved is None:
        conserved = np.zeros(L, bool)
    p_sub = 1.0 - identity
    for i, a in enumerate(anc):
        p = p_sub * (0.2 if conserved[i] else 1.0)
        r = rng.random()
        if r < 0.01:          # deletion
            continue
        if r < 0.02:          # insertion
            out.append(_sample_seq(rng, 1, comp))
        if rng.random() < p:
            out.append(_sample_seq(rng, 1, comp))
        else:
            out.append(a)
    return "".join(out)


def make_family(rng, ancestor, n, id_lo, id_hi, comp, conserved_frac=0.3):
    conserved = rng.random(len(ancestor)) < conserved_frac
    members = []
    for _ in range(n):
        ident = rng.uniform(id_lo, id_hi)
        members.append(mutate(rng, ancestor, ident, comp, conserved))
    return members


def build_dataset(seed=0, n_benign_fams=40, n_per_benign=150, n_S=560, n_H=400):
    rng = np.random.default_rng(seed)
    sup_len = int(rng.integers(130, 170))
    sup_comp = _family_comp(rng)
    super_anc = _sample_seq(rng, sup_len, sup_comp)
    # S and H: ~55% identity to the super-ancestor each (=> ~30-35% to each other)
    H_comp = 0.5 * sup_comp + 0.5 * _family_comp(rng)       # H shares only half of S's composition bias
    H_comp = H_comp / H_comp.sum()
    S_anc = mutate(rng, super_anc, 0.55, sup_comp)
    H_anc = mutate(rng, super_anc, 0.55, H_comp)
    S = make_family(rng, S_anc, n_S, 0.60, 0.92, sup_comp)
    H = make_family(rng, H_anc, n_H, 0.60, 0.92, H_comp)
    benign = {}
    for f in range(n_benign_fams):
        L = int(rng.integers(100, 200))
        comp = _family_comp(rng)
        anc = _sample_seq(rng, L, comp)
        benign[f"B{f:02d}"] = make_family(rng, anc, n_per_benign, 0.55, 0.92, comp)
    # splits
    S = list(S)
    rng.shuffle(S)
    splits = {
        "S_train": S[:360],        # defender (probe / LoRA) + LM pretraining
        "S_calib": S[360:400],     # frozen-threshold calibration (never trained on)
        "S_attacker": S[400:430],  # public records the attacker knows (also in LM pretraining data)
        "S_oracle": S[430:480],    # references for the family-membership oracle (unseen by LM)
        "S_test": S[480:560],      # in-family test set (unseen)
    }
    H = list(H)
    splits["H_train"] = H[:200]    # LM pretraining only -- NEVER used by any defense
    splits["H_attacker"] = H[200:230]
    splits["H_oracle"] = H[230:280]
    splits["H_test"] = H[280:400]
    btrain, bcalib, btest = [], [], []
    bfam = {}
    for k, v in benign.items():
        v = list(v)
        rng.shuffle(v)
        btrain += v[:100]
        bcalib += v[100:125]
        btest += v[125:]
        bfam[k] = v
    splits["B_train"], splits["B_calib"], splits["B_test"] = btrain, bcalib, btest
    return splits, dict(S_anc=S_anc, H_anc=H_anc)


if __name__ == "__main__":
    sp, _ = build_dataset(0)
    for k, v in sp.items():
        print(k, len(v), "mean len", sum(map(len, v)) / len(v))
