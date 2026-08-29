"""Single-axis paraphrase generation (13 LIBERO-Para taxonomy cells).

Reads:
  - --input_csv with columns `key, instruction` (one row per base task)
  - --type_info_csv with columns `idx, high, mid, low` (paraphrase cells)

Writes (under --output_dir):
  - llm_generated_<timestamp>.csv  (raw LLM output)
  - llm_verified_<timestamp>.csv   (verifier-accepted subset)

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

from prompt import general_prompt, type_prompts, verifier_prompt


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


def generate_paraphrase(api_key, model, instruction, task_prompt, num_para):
    prompt = (
        f"{general_prompt}\n{task_prompt}\n{instruction}\n"
        f"Generate {num_para} paraphrases. Please output with the following format and "
        f"only output your answer.\nthe format: 1. instr\n2. instr ...\nout:"
    )
    return api_request_with_retry(api_key, model, prompt)


def verify_paraphrases(api_key, model, task_prompt, original_instruction, paraphrases_list):
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(paraphrases_list))
    prompt = f"""{verifier_prompt}

    TASK SPECIFICATION:
    {task_prompt}

    ORIGINAL INSTRUCTION:
    {original_instruction}

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
        # strip an LLM intro like "out:", "Output -", "Here are the paraphrases:"
        # (may appear on its own line OR inline before the first numbered item)
        line = _LEAD_NOISE_RE.sub("", line)
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-\*•]\s*", "", cleaned)
        cleaned = cleaned.strip("\"'`")
        # strip wrapping markdown bold/italic
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
    ap.add_argument("--input_csv", required=True,
                    help="CSV with columns key, instruction")
    ap.add_argument("--type_info_csv", default=str(Path(__file__).with_name("each_type_info.csv")))
    ap.add_argument("--output_dir", default=str(Path(__file__).with_name("llm_output")))
    ap.add_argument("--num_paraphrases", type=int, default=3)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("OPENROUTER_API_KEY is required (env var or --api_key)")

    instructions_df = pd.read_csv(args.input_csv)
    type_info_df = pd.read_csv(args.type_info_csv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"llm_generated_{timestamp}.csv"

    cols = ["high", "mid", "low", "eval", "batch_idx", "new_instruction", "original_instruction"]
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)

    for _, type_row in type_info_df.iterrows():
        high, mid, low = type_row["high"], type_row["mid"], type_row["low"]
        task_key = f"{high}_{mid}_{low}"
        task_prompt = type_prompts.get(task_key, "")
        if not task_prompt:
            print(f"WARN: no prompt for {task_key}")
            continue

        for _, inst_row in instructions_df.iterrows():
            instruction = inst_row["instruction"]
            eval_idx = inst_row["key"]

            try:
                resp = generate_paraphrase(
                    args.api_key, args.model, instruction, task_prompt, args.num_paraphrases
                )
            except Exception as e:
                print(f"  generation failed for {eval_idx} {task_key}: {e}")
                continue

            paraphrases = parse_paraphrases(resp)
            batch = [
                {
                    "high": high, "mid": mid, "low": low,
                    "eval": eval_idx, "batch_idx": i,
                    "new_instruction": p, "original_instruction": instruction,
                }
                for i, p in enumerate(paraphrases)
            ]
            if batch:
                pd.DataFrame(batch).to_csv(output_path, mode="a", header=False, index=False)
                print(f"  saved {len(batch)} for {eval_idx} ({task_key})")

    print(f"\nraw saved -> {output_path}")

    # ---- verification ----
    print("\nverifying ...")
    generated_df = pd.read_csv(output_path)
    verified_path = out_dir / f"llm_verified_{timestamp}.csv"
    pd.DataFrame(columns=cols).to_csv(verified_path, index=False)

    grouped = generated_df.groupby(["high", "mid", "low", "eval", "original_instruction"])
    for (high, mid, low, eval_idx, orig), group in grouped:
        task_key = f"{high}_{mid}_{low}"
        task_prompt = type_prompts.get(task_key, "")
        if not task_prompt:
            continue
        with_idx = list(zip(group["new_instruction"].tolist(), group["batch_idx"].tolist()))
        plist = [p for p, _ in with_idx]
        try:
            accepted = verify_paraphrases(args.api_key, args.model, task_prompt, orig, plist)
        except Exception as e:
            print(f"  verify failed for {eval_idx} {task_key}: {e}")
            continue
        rows = []
        for ap_ in accepted:
            for p, oi in with_idx:
                if p == ap_:
                    rows.append({
                        "high": high, "mid": mid, "low": low,
                        "eval": eval_idx, "batch_idx": oi,
                        "new_instruction": ap_, "original_instruction": orig,
                    })
                    break
        if rows:
            pd.DataFrame(rows).to_csv(verified_path, mode="a", header=False, index=False)
            print(f"  verified {len(rows)}/{len(plist)} for {eval_idx} ({task_key})")

    print(f"\nverified saved -> {verified_path}")


if __name__ == "__main__":
    main()
