"""Shared constants and helpers for the ProtGuard pipeline-validation study."""
import random
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA)
# approximate UniProt background amino-acid frequencies
BG = {"A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.37, "Q": 3.93, "E": 6.75, "G": 7.07,
      "H": 2.27, "I": 5.96, "L": 9.66, "K": 5.84, "M": 2.42, "F": 3.86, "P": 4.70, "S": 6.56,
      "T": 5.34, "W": 1.08, "Y": 2.92, "V": 6.87}
BG_VEC = np.array([BG[a] for a in AA], dtype=float)
BG_VEC /= BG_VEC.sum()
EOS = "<|endoftext|>"

NONCANON = "XBZUOJ"
# visually confusable (Cyrillic / Greek) look-alikes for canonical letters
HOMOGLYPH = {"A": "А", "C": "С", "E": "Е", "H": "Н", "K": "К",
             "M": "М", "P": "Р", "S": "Ѕ", "T": "Т", "Y": "Ү",
             "I": "І", "V": "Ѵ", "W": "Ԝ"}
ZWSP = "​"


def canon_only(s: str) -> str:
    """Keep only canonical residues (used to turn model text output into a residue string)."""
    return "".join(c for c in s if c in AA_SET)


def rng_of(seed):
    return random.Random(seed), np.random.default_rng(seed)
