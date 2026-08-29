"""Compositional paraphrase generation (3 obj × 10 act = 30 cells).

Reads the verified CSV from main_single.py (must contain both `obj` and `act`
high types) and produces merged paraphrases that combine BOTH transformations.

Writes (under --output_dir):
  - llm_merged_<timestamp>.csv          (raw merged output)
  - llm_merged_verified_<timestamp>.csv (verifier-accepted subset)

API key is read from --api_key OR the OPENROUTER_API_KEY env var.
"""
import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException, Timeout

from prompt import merge_prompt, merge_verifier_prompt


def api_request_with_retry(api_key, model, prompt, max_retries=3, timeout=120):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (ChunkedEncodingError, ConnectionError, Timeout) as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"  retry {attempt+1}/{max_retries}: {type(e).__name__}; sleeping {wait}s")
                time.sleep(wait)
            else:
                raise
        except RequestException:
            raise


def generate_merged(api_key, model, original, obj_examples, act_examples, obj_low, act_low, num_para):
    obj_str = "\n".join(f"- {ex}" for ex in obj_examples)
    act_str = "\n".join(f"- {ex}" for ex in act_examples)
    prompt = f"""{merge_prompt}

ORIGINAL INSTRUCTION:
{original}

OBJECT PARAPHRASE EXAMPLES (low={obj_low}):
{obj_str}

ACTION PARAPHRASE EXAMPLES (low={act_low}):
{act_str}

Generate {num_para} merged paraphrases that combine BOTH object changes AND action changes.
Output one merged paraphrase per line, without numbering or additional text.
"""
    return api_request_with_retry(api_key, model, prompt)


def verify_merged(api_key, model, original, obj_examples, act_examples, obj_low, act_low, paraphrases_list):
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(paraphrases_list))
    obj_str = "\n".join(f"- {ex}" for ex in obj_examples)
    act_str = "\n".join(f"- {ex}" for ex in act_examples)
    prompt = f"""{merge_verifier_prompt}

TASK SPECIFICATION:
This is a MERGED paraphrase task combining:
- Object paraphrase type: {obj_low}
- Action paraphrase type: {act_low}

The paraphrases MUST have BOTH:
1. Object changes following the pattern in object examples
2. Action changes following the pattern in action examples

ORIGINAL INSTRUCTION:
{original}

OBJECT PARAPHRASE EXAMPLES (for reference):
{obj_str}

ACTION PARAPHRASE EXAMPLES (for reference):
{act_str}

PARAPHRASES TO VERIFY:
{numbered}

Please output ONLY the ACCEPTED paraphrases, one per line, without any numbering or additional text.
"""
    resp = api_request_with_retry(api_key, model, prompt)
    accepted = []
    for line in resp.strip().split("\n"):
        line = line.strip()
        if line and line in paraphrases_list:
            accepted.append(line)
    return accepted


_LEAD_NOISE_RE = re.compile(
    r"^\s*(?:out|output|paraphrases?|here(?:'s| are| is)?[^:]*)\s*[:\-]\s*",
    re.I,
)


def parse_paraphrases(text):
    out = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = _LEAD_NOISE_RE.sub("", line)
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-\*•]\s*", "", cleaned)
        cleaned = cleaned.strip("\"'`")
        cleaned = re.sub(r"^\*+|\*+$", "", cleaned).strip()
        if cleaned.endswith(".") and "?" not in cleaned:
            cleaned = cleaned[:-1]
        if cleaned:
            out.append(cleaned)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--model", default="google/gemini-2.5-pro")
    ap.add_argument("--verified_csv", required=True,
                    help="output of main_single.py (llm_verified_*.csv)")
    ap.add_argument("--output_dir", default=str(Path(__file__).with_name("llm_output")))
    ap.add_argument("--num_paraphrases", type=int, default=3)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("OPENROUTER_API_KEY is required (env var or --api_key)")

    df = pd.read_csv(args.verified_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"llm_merged_{timestamp}.csv"

    cols = ["high", "mid", "low", "eval", "batch_idx", "new_instruction", "original_instruction"]
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)

    unique_evals = df[["eval", "original_instruction"]].drop_duplicates()
    obj_lows = df[df["high"] == "obj"]["low"].unique()
    act_lows = df[df["high"] == "act"]["low"].unique()

    print(f"obj_lows ({len(obj_lows)}): {list(obj_lows)}")
    print(f"act_lows ({len(act_lows)}): {list(act_lows)}")
    print(f"evals: {len(unique_evals)}")

    for obj_low in obj_lows:
        for act_low in act_lows:
            print(f"\n[{obj_low} + {act_low}]")
            for _, ev_row in unique_evals.iterrows():
                eval_idx = ev_row["eval"]
                orig = ev_row["original_instruction"]
                obj_data = df[(df["high"] == "obj") & (df["low"] == obj_low) & (df["eval"] == eval_idx)]
                act_data = df[(df["high"] == "act") & (df["low"] == act_low) & (df["eval"] == eval_idx)]
                if obj_data.empty or act_data.empty:
                    print(f"  skip {eval_idx}: missing examples")
                    continue
                obj_ex = obj_data["new_instruction"].tolist()
                act_ex = act_data["new_instruction"].tolist()
                obj_mid = obj_data["mid"].iloc[0]
                act_mid = act_data["mid"].iloc[0]
                try:
                    resp = generate_merged(
                        args.api_key, args.model, orig,
                        obj_ex, act_ex, obj_low, act_low, args.num_paraphrases,
                    )
                except Exception as e:
                    print(f"  gen failed {eval_idx}: {e}")
                    continue
                merged = parse_paraphrases(resp)
                rows = [
                    {
                        "high": "merged",
                        "mid": f"{obj_mid}+{act_mid}",
                        "low": f"{obj_low}+{act_low}",
                        "eval": eval_idx, "batch_idx": i,
                        "new_instruction": p, "original_instruction": orig,
                    }
                    for i, p in enumerate(merged)
                ]
                if rows:
                    pd.DataFrame(rows).to_csv(output_path, mode="a", header=False, index=False)
                    print(f"  saved {len(rows)} for {eval_idx}")

    print(f"\nraw merged saved -> {output_path}")

    # ---- verification ----
    print("\nverifying ...")
    merged_df = pd.read_csv(output_path)
    verified_path = out_dir / f"llm_merged_verified_{timestamp}.csv"
    pd.DataFrame(columns=cols).to_csv(verified_path, index=False)

    grouped = merged_df.groupby(["mid", "low", "eval", "original_instruction"])
    for (mid_combined, low_combined, eval_idx, orig), group in grouped:
        obj_low, act_low = low_combined.split("+")
        obj_ex = df[(df["high"] == "obj") & (df["low"] == obj_low) & (df["eval"] == eval_idx)]["new_instruction"].tolist()
        act_ex = df[(df["high"] == "act") & (df["low"] == act_low) & (df["eval"] == eval_idx)]["new_instruction"].tolist()
        with_idx = list(zip(group["new_instruction"].tolist(), group["batch_idx"].tolist()))
        plist = [p for p, _ in with_idx]
        try:
            accepted = verify_merged(args.api_key, args.model, orig, obj_ex, act_ex, obj_low, act_low, plist)
        except Exception as e:
            print(f"  verify failed {eval_idx}: {e}")
            continue
        rows = []
        for ap_ in accepted:
            for p, oi in with_idx:
                if p == ap_:
                    rows.append({
                        "high": "merged",
                        "mid": "comp",
                        "low": low_combined,
                        "eval": eval_idx, "batch_idx": oi,
                        "new_instruction": ap_, "original_instruction": orig,
                    })
                    break
        if rows:
            pd.DataFrame(rows).to_csv(verified_path, mode="a", header=False, index=False)
            print(f"  verified {len(rows)}/{len(plist)} for {eval_idx} ({low_combined})")

    print(f"\nverified merged saved -> {verified_path}")


if __name__ == "__main__":
    main()
