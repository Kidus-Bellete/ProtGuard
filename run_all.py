"""Step 2: build arms (thresholds frozen), run benign evaluation, RQ1 + RQ2 red-team, save results.json."""
import sys, json, os, time
import numpy as np
import jax
import model as M
import defenses as Df
from attacker import Attacker, FAMILIES
from oracle import Oracle, quality_ok
from common import canon_only

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_STATIC = int(sys.argv[2]) if len(sys.argv) > 2 else 2
PER_ROUND = int(sys.argv[3]) if len(sys.argv) > 3 else 500
RR_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
RES = os.path.join(M.OUT, "results.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    sp = json.load(open(os.path.join(M.OUT, "splits.json")))
    rs = np.random.default_rng(42)          # same membership as in LM training; shuffle so subsets span all families
    for k in ["B_train", "B_calib", "B_test"]:
        rs.shuffle(sp[k])
    tok = M.load_tok()
    P = M.load_params(os.path.join(M.OUT, "base.npz"), tok.get_vocab_size())
    orc_cfg = json.load(open(os.path.join(M.OUT, "oracle.json")))
    orS = Oracle(sp["S_oracle"]); orS.T = orc_cfg["T_S"]
    orH = Oracle(sp["H_oracle"]); orH.T = orc_cfg["T_H"]
    rng = np.random.default_rng(0)
    arms, info = Df.build_arms(P, tok, sp, rng, log=log, rr_steps=RR_STEPS)
    results = dict(info={k: v for k, v in info.items() if k != "steer_grid"}, oracle=orc_cfg,
                   base_ppl=json.load(open(os.path.join(M.OUT, "base_ppl.json"))))
    results["info"]["steer_grid"] = {str(k): v for k, v in info["steer_grid"].items()}
    ntok = lambda s: Df.n_tokens(tok, s)
    key = jax.random.PRNGKey(123)

    # ---------------- benign evaluation: FPR, utility ----------------
    log("benign evaluation")
    benign_q = [s[:15] for s in sp["B_test"][::2][:400]]                    # canonical benign requests
    ben = {}
    for name, arm in arms.items():
        key, k = jax.random.split(key)
        out = arm.query(benign_q, k)
        blocked = np.mean([o["blocked"] for o in out])
        okq = np.mean([(not o["blocked"]) and quality_ok(o["res"]) for o in out])
        reasons = {}
        for o in out:
            if o["blocked"]:
                reasons[o["reason"]] = reasons.get(o["reason"], 0) + 1
        ppl = M.perplexity(arm.P, tok, sp["B_test"][:300], steer=arm.steer)
        ben[name] = dict(fpr=float(blocked), quality_pass=float(okq), ppl=ppl, reasons=reasons)
        log(name, ben[name])
    results["benign"] = ben

    # ---------------- generalisation (RQ2): static probe detection on natural sequences ----------------
    log("static probe detection on natural sequences")
    gen = {}
    for name in ["C", "D", "E"]:
        arm = arms[name]
        gen[name] = dict(
            S_test=Df.detect_rate(arm.P, tok, arm.probe, sp["S_test"], steer=arm.steer),
            H_test=Df.detect_rate(arm.P, tok, arm.probe, sp["H_test"][:100], steer=arm.steer),
            benign=Df.detect_rate(arm.P, tok, arm.probe, sp["B_test"][:300], steer=arm.steer))
        log(name, gen[name])
    results["detect"] = gen
    json.dump(results, open(RES, "w"), indent=1)

    # ---------------- red-teaming ----------------
    benign_pool = sp["B_test"][:200]

    def campaign(target, mode, seeds):
        known = sp["S_attacker"] if target == "S" else sp["H_attacker"]
        orc = orS if target == "S" else orH
        out = {a: [] for a in arms}
        for sd in seeds:
            for name, arm in arms.items():
                atk = Attacker(known, benign_pool, mode=mode, seed=1000 * (1 + sd),
                               per_round=PER_ROUND)
                t0 = time.time()
                key_ = jax.random.PRNGKey(5000 + sd)
                pr = atk.run(arm, orc, key_, ntok)
                out[name].append(pr)
                log(f"target={target} mode={mode} seed={sd} arm={name} evasion/round="
                    f"{[round(r['evasion'], 3) for r in pr]} allowed={[round(r['allowed'], 2) for r in pr]} "
                    f"({time.time()-t0:.0f}s)")
            results[f"red_{target}_{mode}"] = out
            json.dump(results, open(RES, "w"), indent=1)
        return out

    campaign("S", "adaptive", range(N_RUNS))
    campaign("S", "static", range(N_STATIC))
    campaign("H", "adaptive", range(max(2, N_RUNS // 2)))
    log("done")


if __name__ == "__main__":
    main()
