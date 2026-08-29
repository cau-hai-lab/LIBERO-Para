# Paraphrase Generation

LLM-based paraphrase generator + verifier, adapted from the LIBERO-Para pipeline for CALVIN.
Produces the 1,935 paraphrases behind [CALVIN-Para](../calvin-para/).

## Taxonomy

Each paraphrase belongs to one of **43 cells** on two axes:

- **Object axis** (4): `none`, `ad` (addition/deletion), `spc` (same-polarity contextual), `sph` (same-polarity habitual)
- **Action axis** (11): `none`, `ad`, `spc`, `sph`, `coord`, `subord`, `need`, `embed`, `perm`, `ques`, `hint`

43 = 13 single-axis (3 obj-only + 10 act-only, both-`none` is the canonical instruction and
is not a cell) + 30 compositional (3 obj × 10 act).

## Output

**15 CALVIN tasks × 43 cells × 3 paraphrases = 1,935.**

The task list is `inputs/calvin_base15_tasks.txt`, the subset of CALVIN tasks that can run
as standalone single-task episodes. See [CALVIN-Para's README](../calvin-para/README.md#task-selection)
for how 34 CALVIN tasks were narrowed to these 15.

| file | contents |
|---|---|
| `paraphrases.json` | the 1,935 paraphrases with full metadata |
| `paraphrase_summary.csv` | per-cell and per-task counts |

Each entry:

```json
{
  "paraphrase_id": "p_00001",
  "base_task_id": "close_drawer",
  "base_original": "push the handle to close the drawer",
  "object_type": "ad",
  "action_type": "ad",
  "cell_id": "obj_ad__act_ad",
  "paraphrase_index": 1,
  "paraphrase": "carefully push the handle to close the storage drawer"
}
```

## Reproducing

```bash
export OPENROUTER_API_KEY=...      # https://openrouter.ai

# 1. Build input CSV (key, instruction) from the CALVIN annotations YAML
./run_01_build_input.sh

# 2. Single-axis cells (13 × 3 per task)
./run_02_single.sh
#    -> llm_output/llm_verified_<TS>.csv

# 3. Compositional cells (30 × 3 per task)
VERIFIED_CSV=llm_output/llm_verified_<TS>.csv ./run_03_compositional.sh
#    -> llm_output/llm_merged_verified_<TS>.csv

# 4. Merge, cap to 3 per cell, emit final JSON + summary
SINGLE_CSV=llm_output/llm_verified_<TS>.csv \
MERGED_CSV=llm_output/llm_merged_verified_<TS>.csv \
./run_04_build_json.sh
```

Steps 2 through 4 write into `llm_output/`, which is gitignored. Regenerating produces a fresh
batch rather than reproducing `paraphrases.json` verbatim (LLM sampling is not
deterministic). The committed `paraphrases.json` is the exact set used in the paper.

## Files

| file | role |
|---|---|
| `prompt.py` | paraphrase + verifier prompt templates (CALVIN scene objects baked in) |
| `build_input_csv.py` | pulls canonical instructions from `new_playtable_validation.yaml` |
| `main_single.py` | generate + verify single-axis paraphrases |
| `main_compositional.py` | generate + verify obj×act compositional paraphrases |
| `build_paraphrases_json.py` | merge, cap to 3 per cell, emit final JSON + summary |
| `verify_only.py` | re-run the verifier over an existing CSV |
| `post_clean.py` | strip LLM artifacts (`out: 1.`, bullets, markdown bold) from a batch |
| `merge_and_review.py` | merge act/obj batches, filter to the 15 tasks, print for review |
| `each_type_info.csv` | 13 single-axis cell definitions |
| `inputs/calvin_base15_tasks.txt` | the 15 published task ids (pipeline default) |
| `inputs/calvin_base20_tasks.txt` | the pre-feasibility-filter 20-task candidate list |
| `inputs/calvin_instructions_15.csv` | canonical instruction per task, built by step 1 |

## Notes

- Default model is Gemini 2.5 Pro via OpenRouter; override with the `MODEL` env var.
- The API key is read from `OPENROUTER_API_KEY` only, never hardcoded.
- Default `--num_paraphrases 3` per cell; raise it for margin before the verifier trims.
- All scripts resolve paths relative to themselves; no absolute paths.
