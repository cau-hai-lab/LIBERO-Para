"""Generic model summary - works for any model with both baseline + paraphrase results."""
import sys, glob, json, os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CALVIN_PARA = Path(__file__).resolve().parents[2]

PARA_EVAL = str(CALVIN_PARA / "paraphrase_eval")
RESULTS_DIR = str(CALVIN_PARA / "RESULTS")

def load_episodes(model, kind, seed="seed7"):
    eps = []
    for f in sorted(glob.glob(os.path.join(PARA_EVAL, "results", model, f"{kind}_{seed}", "eval_*.json"))):
        d = json.load(open(f))
        for e in d["episodes"]:
            eps.append({"task_id": d["task_id"], "success": bool(e.get("success", False)),
                       "metadata": e.get("metadata", {}), "lang_override": e.get("lang_override")})
    return eps

sys.path.insert(0, str(CALVIN_PARA.parent / "metrics"))
from compute_pride_score_example import StructuralSimilarity, KeywordSimilarity


def add_similarity(para_eps, alpha=0.5):
    import spacy
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    nlp = spacy.load("en_core_web_sm")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    ss = StructuralSimilarity(nlp); ks = KeywordSimilarity(nlp, encoder)
    pjson = json.load(open(CALVIN_PARA.parent / "paraphrase_generation/paraphrases.json"))
    pmeta = {p["paraphrase_id"]: p for p in pjson}
    cache = {}
    for e in tqdm(para_eps, desc="sim"):
        pid = e["metadata"].get("paraphrase_id")
        meta = pmeta.get(pid) if pid else None
        if not meta:
            e["pd"] = None; continue
        orig, para = meta["base_original"], e["lang_override"]
        if (orig, para) in cache:
            sk, st = cache[(orig, para)]
        else:
            s = ss.compute(orig, para); k = ks.compute(orig, para)
            sk, st = k["keyword_similarity"], s["structural_similarity"]
            cache[(orig, para)] = (sk, st)
        e["pd"] = 1.0 - (alpha * sk + (1 - alpha) * st)
    return para_eps

def compute_pride(eps):
    pds = [e["pd"] for e in eps if e["pd"] is not None]
    succ = [e["pd"] for e in eps if e["pd"] is not None and e["success"]]
    return sum(succ) / sum(pds) * 100 if pds else None

def grouped_obj_sr(eps):
    pres = [e for e in eps if e["metadata"].get("object_type") in ("none","ad")]
    para = [e for e in eps if e["metadata"].get("object_type") in ("spc","sph")]
    return {"pres": np.mean([e["success"] for e in pres])*100, "pres_n": len(pres),
            "para": np.mean([e["success"] for e in para])*100, "para_n": len(para)}

def cat_breakdown(cache_path):
    eps = json.load(open(cache_path))["episodes"]
    cats = Counter(e["category"] for e in eps); n = len(eps); f = n - cats.get("Success", 0)
    return {"n": n, "succ": cats.get("Success", 0), "near": cats.get("NearGT", 0),
            "cross": cats.get("Cross-NearGT", 0), "unm": cats.get("Unmatched", 0),
            "fail": f, "fargt": cats.get("Cross-NearGT", 0) + cats.get("Unmatched", 0)}

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "flower"
    base = load_episodes(model, "baseline")
    para = load_episodes(model, "paraphrase")
    print(f"loaded {model}: baseline={len(base)}, paraphrase={len(para)}")
    csr = np.mean([e["success"] for e in base])*100
    psr = np.mean([e["success"] for e in para])*100
    para = add_similarity(para)
    pride = compute_pride(para)
    grp = grouped_obj_sr(para)
    cb = cat_breakdown(os.path.join(PARA_EVAL, f"analysis/cache/{model}__neargt_fargt_v2.json"))

    print("\n" + "="*70)
    print(f" {model.upper()} — comprehensive summary")
    print("="*70)
    print(f"\nCALVIN canonical SR              : {csr:6.2f}%   (n={len(base)})")
    print(f"CALVIN-Para overall SR           : {psr:6.2f}%   (n={len(para)})")
    print(f"  drop                            : {csr-psr:6.2f}pp")
    print(f"CALVIN-Para PRIDE (alpha=0.5)    : {pride:6.2f}")
    print(f"\n--- obj-axis grouped SR ---")
    print(f"  Object-preserving (None+AD)    : {grp['pres']:6.2f}%   (n={grp['pres_n']})")
    print(f"  Object-paraphrased (SPC+SPH)   : {grp['para']:6.2f}%   (n={grp['para_n']})")
    print(f"  drop                            : {grp['pres']-grp['para']:6.2f}pp")
    print(f"\n--- NearGT/FarGT (cluster-aware v2, n={cb['n']}) ---")
    print(f"  Success                        : {cb['succ']:5d}  ({cb['succ']/cb['n']*100:5.1f}%)")
    print(f"  Failures                       : {cb['fail']:5d}  ({cb['fail']/cb['n']*100:5.1f}%)")
    print(f"  NearGT                         : {cb['near']:5d}  ({cb['near']/cb['n']*100:5.1f}% all  | {cb['near']/cb['fail']*100:5.1f}% fail)")
    print(f"  FarGT total                    : {cb['fargt']:5d}  ({cb['fargt']/cb['n']*100:5.1f}% all  | {cb['fargt']/cb['fail']*100:5.1f}% fail)")
    print(f"    Unmatched                    : {cb['unm']:5d}  ({cb['unm']/cb['n']*100:5.1f}% all  | {cb['unm']/cb['fail']*100:5.1f}% fail)")
    print(f"    Cross-NearGT                 : {cb['cross']:5d}  ({cb['cross']/cb['n']*100:5.1f}% all  | {cb['cross']/cb['fail']*100:5.1f}% fail)")
    rows = [
        {"metric": "CALVIN canonical SR (%)", "value": round(csr,2)},
        {"metric": "CALVIN-Para overall SR (%)", "value": round(psr,2)},
        {"metric": "CALVIN-Para PRIDE (alpha=0.5)", "value": round(pride,2)},
        {"metric": "Obj-preserving SR (%)", "value": round(grp['pres'],2)},
        {"metric": "Obj-paraphrased SR (%)", "value": round(grp['para'],2)},
        {"metric": "NearGT % of all", "value": round(cb['near']/cb['n']*100,2)},
        {"metric": "NearGT % of failures", "value": round(cb['near']/cb['fail']*100,2)},
        {"metric": "FarGT total % of all", "value": round(cb['fargt']/cb['n']*100,2)},
        {"metric": "FarGT total % of failures", "value": round(cb['fargt']/cb['fail']*100,2)},
        {"metric": "Unmatched % of failures", "value": round(cb['unm']/cb['fail']*100,2)},
        {"metric": "Cross-NearGT % of failures", "value": round(cb['cross']/cb['fail']*100,2)},
    ]
    out = os.path.join(RESULTS_DIR, f"{model}__summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote -> {out}")

if __name__ == "__main__":
    main()
