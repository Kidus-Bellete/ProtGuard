# ProtGuard pipeline (stand-in model, real Swiss-Prot data)
Real-data run order (PG_OUT=artifacts_real):
  python data_real.py      # needs sprot.pkl (parsed uniprot_sprot.fasta.gz; not included, 500 MB+)
  python train_base.py 1800 ; python run_all.py 4 2 500 300 ; python extra_checks.py ; python sweep_steer.py ; python report.py
Synthetic-family run (earlier validation) is in artifacts_synthetic/ (data.py).
Files: data_real.py (Swiss-Prot families + cluster-aware splits), model.py (JAX GPT-2 stand-in: KV-cache generation, L2 mask, L3 probe, steering, LoRA),
defenses.py (L1-L4, arms A-E + ablations C0/D0/E0, frozen thresholds), attacker.py (pre-registered variation families, Thompson-sampling adaptation),
oracle.py (family-membership oracle, ProtParam checks), run_all.py / extra_checks.py / sweep_steer.py / report.py.
To run on real ProtGPT2: replace model.py's network/generate with the HF checkpoint (same interface: prefill/step, hidden states at a layer, steering hook, PEFT LoRA).

## Data (not in the repo)
Download `uniprot_sprot.fasta.gz` from UniProt, parse to `sprot.pkl` (list of dicts: acc, entry, name, org, ox, seq; canonical-alphabet records only), then run `data_real.py`.

## Results
See `RESULTS_REAL.md` (real Swiss-Prot, stand-in model) and `RESULTS_SYNTHETIC.md` (earlier synthetic validation).
