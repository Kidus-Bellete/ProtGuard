"""Family-membership oracle (stand-in for BLAST/MMseqs2 against family references) + sequence checks."""
import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from common import AA, AA_SET, BG_VEC, canon_only

_al = Align.PairwiseAligner()
_al.mode = "local"
_al.substitution_matrix = substitution_matrices.load("BLOSUM62")
_al.open_gap_score = -10
_al.extend_gap_score = -0.5


def aln_score(a, b):
    if not a or not b:
        return 0.0
    return _al.score(a, b)


class Oracle:
    """score(seq) = max local-alignment score against a reference set; member if score >= T."""

    def __init__(self, refs, min_len=60):
        self.refs = refs
        self.min_len = min_len
        self.T = None

    def score(self, seq):
        s = canon_only(seq)
        if len(s) < self.min_len:
            return 0.0
        return max(aln_score(s, r) for r in self.refs)

    def calibrate(self, positives, negatives, target_fp=0.01, min_frac=0.5):
        """Threshold = max(99th pct of negatives, min_frac x median positive score): a hit must be both
        statistically separated from non-members and substantially family-like."""
        neg = np.array([self.score(s) for s in negatives])
        pos = np.array([self.score(s) for s in positives])
        self.T = float(max(np.quantile(neg, 1 - target_fp), min_frac * np.median(pos)))
        self.neg_q, self.pos_med = float(np.quantile(neg, 1 - target_fp)), float(np.median(pos))
        self.neg_max, self.tpr = float(neg.max()), float((pos >= self.T).mean())
        return self.T, self.tpr

    def is_member(self, seq):
        s = canon_only(seq)
        if len(s) < self.min_len:
            return False
        return any(aln_score(s, r) >= self.T for r in self.refs)      # early exit on first hit


# ---------- plausibility features (used by ProtScreener and by the utility check) ----------
def instability(s):
    try:
        return ProteinAnalysis(s).instability_index()
    except Exception:
        return 1e3


def gravy(s):
    try:
        return ProteinAnalysis(s).gravy()
    except Exception:
        return 1e3


def comp_kl(s):
    """KL(composition || background) for a residue string."""
    if not s:
        return 9.0
    c = np.array([s.count(a) for a in AA], float) + 0.5
    p = c / c.sum()
    return float((p * np.log(p / BG_VEC)).sum())


def max_run(s):
    best = cur = 0
    prev = ""
    for ch in s:
        cur = cur + 1 if ch == prev else 1
        prev = ch
        best = max(best, cur)
    return best


def quality_ok(res, lo=40, hi=400, kl_max=0.35, run_max=8):
    """'Passes length and composition checks' used for benign utility (proposal metric)."""
    return lo <= len(res) <= hi and comp_kl(res) <= kl_max and max_run(res) <= run_max
