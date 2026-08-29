"""Check which of our candidate tasks can appear as the first task in a
randomly sampled CALVIN sequence (i.e. usable for single-task eval).

Samples N base sequences and counts first-task occurrences for each
candidate. Tasks that get >= TRIALS_PER_TASK are usable for the single-task
eval; the rest must be dropped or set up manually.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "calvin/calvin_models"))
from calvin_agent.evaluation.multistep_sequences import get_sequences

CANDIDATES = [
    "open_drawer", "close_drawer",
    "move_slider_left", "move_slider_right",
    "turn_on_lightbulb", "turn_off_lightbulb",
    "turn_on_led", "turn_off_led",
    "lift_red_block_table", "lift_blue_block_table",
    "lift_red_block_slider", "lift_red_block_drawer",
    "place_in_slider", "place_in_drawer",
    "push_red_block_right", "push_red_block_left",
    "rotate_red_block_left", "rotate_red_block_right",
    "stack_block", "unstack_block",
]
N_BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
TRIALS_PER_TASK = 20

print(f"sampling {N_BASE} base sequences ...")
seqs = get_sequences(N_BASE)
print(f"  got {len(seqs)} sequences")

first_counts = Counter()
for _, seq in seqs:
    first_counts[str(seq[0])] += 1

print(f"\n{'task':<28} {'first-slot count':>16}  {'usable?':>8}")
print("-" * 58)
usable, marginal, unusable = [], [], []
for t in CANDIDATES:
    n = first_counts.get(t, 0)
    if n >= TRIALS_PER_TASK:
        tag = "OK"
        usable.append(t)
    elif n > 0:
        tag = "few"
        marginal.append((t, n))
    else:
        tag = "NONE"
        unusable.append(t)
    print(f"  {t:<26} {n:>16}  {tag:>8}")

print(f"\nusable ({len(usable)}): {usable}")
print(f"marginal (<{TRIALS_PER_TASK}): {marginal}")
print(f"unusable (0 occurrences): {unusable}")
