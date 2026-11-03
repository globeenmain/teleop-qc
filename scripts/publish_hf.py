"""Publish the eval set and a judge's verdicts as a Hugging Face dataset.

    python scripts/publish_hf.py --verdicts results/judge_claude-opus-5-5.jsonl \
        --grids runs/claude-opus-5-5 --repo 2skill3d4u/droid-qc-eval

One row per episode: the frame grid the judge saw, the human label, the instruction (which the judge never sees),
the pointers into lerobot/droid_1.0.1, and the judge's verdict. The frozen eval set, the raw verdicts and the
judge's prompt and answer schema are uploaded next to it as files. Reads HF_TOKEN; prints no credential.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from teleop_qc.cli import load_evalset, report
from teleop_qc.judges import SCHEMA, SYSTEM

ROOT = Path(__file__).resolve().parent.parent


def rows(evalset: Path, verdicts: Path, grids: Path) -> list[dict]:
    judged = {r["episode_index"]: r for r in map(json.loads, verdicts.read_text().splitlines())}
    out = []
    for episode in load_evalset(evalset):
        v = judged[episode.index]
        out.append({
            "episode_index": episode.index,
            "grid": str(grids / f"{episode.index:06d}.jpg"),
            "human_success": episode.success,
            "instruction": episode.task,
            "frames": episode.length,
            "seconds": episode.seconds,
            "videos": json.dumps([asdict(s) for s in episode.videos]),
            "judge_model": v["model"],
            "judge_success": v["success"],
            "judge_confidence": v["confidence"],
            "judge_p_success": v["p_success"],
            "judge_failure_mode": v["failure_mode"],
            "judge_evidence": v["evidence"],
            "judge_attempts": v["attempts"],
            "judge_input_tokens": v["input_tokens"],
            "judge_output_tokens": v["output_tokens"],
        })
    return out


def card(table: pd.DataFrame, metrics: dict, model: str) -> str:
    n, successes = len(table), int(table.human_success.sum())
    lo, hi = metrics["auroc_ci"]
    return f"""---
license: apache-2.0
pretty_name: DROID QC eval set
task_categories:
- video-classification
- image-classification
tags:
- robotics
- teleoperation
- data-quality
- vlm-as-a-judge
- calibration
- droid
- lerobot
size_categories:
- n<1K
source_datasets:
- lerobot/droid_1.0.1
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*.parquet
---

# DROID QC eval set

{n} teleoperation episodes from [DROID 1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1)
({successes} labelled successful, {n - successes} failed by the people who collected them), the frame grid a
vision-language judge sees for each, and the verdicts of `{model}`. Built with
[teleop-qc](https://github.com/globeenmain/teleop-qc), which measures how far a VLM judge of robot task success can
be trusted before it replaces human review in a data pipeline.

## What is in a row

| column | meaning |
|---|---|
| `episode_index` | episode in `lerobot/droid_1.0.1` |
| `grid` | the image the judge sees (968x1100 JPEG): 3 cameras in columns, 6 moments in rows, first at the top |
| `human_success` | DROID's success label, set by the operator who collected the episode |
| `instruction` | DROID's language instruction, `""` when missing. **Never shown to the judge** (see below) |
| `frames`, `seconds` | episode length at 15 fps |
| `videos` | where each camera stream lives in the DROID video files (JSON) |
| `judge_*` | the verdict: success, confidence (0.5 to 1), `p_success`, failure mode, evidence, attempts, tokens |

Columns of the grid, left to right: exterior camera 1, exterior camera 2, wrist camera. `judge_p_success` is the
confidence for a success verdict and 1 - confidence for a failure. The judge abstains instead of guessing when
its answer fails a schema and consistency check twice: abstentions have null `judge_success` and
`judge_p_success`.

## How the episodes were drawn

{n // 2} successful and {n // 2} failed episodes, drawn with seed 0 from the first metadata file of DROID 1.0.1
(14,153 episodes). The set is frozen as `evalset/droid_evalset_v1.jsonl`; a change makes a new version.

**Why the judge never sees the instruction.** In that metadata file, 94.6% of the episodes labelled successful
have a language instruction, against 0.3% of the failed ones: the instructions were crowdsourced after collection,
mostly for successful episodes. A judge shown the instruction could read the label off its presence.

## Results of `{model}`

Command: `teleop-qc judge --evalset evals/droid_evalset_v1.jsonl --out runs/{model}.jsonl` (effort `medium`,
structured outputs), then `teleop-qc report`.

| metric | value |
|---|---|
| episodes judged | {metrics["n"]:.0f} of {metrics["episodes"]} ({metrics["abstained"]} abstentions) |
| accuracy at p = 0.5 | {metrics["accuracy"]:.3f} |
| Cohen's kappa | {metrics["kappa"]:.3f} |
| AUROC (95% bootstrap interval) | {metrics["auroc"]:.3f} ({lo:.3f} to {hi:.3f}) |
| Brier score | {metrics["brier"]:.3f} |
| ECE (10 uniform bins) | {metrics["ece"]:.3f} |
| ECE after cross-fitted isotonic recalibration | {metrics["ece_after_isotonic"]:.3f} |
| share accepted without review at 95% precision | {metrics["auto_accept_at_95"]:.3f} |
| share rejected without review at 90% precision | {metrics["auto_reject_at_90"]:.3f} |
| cost per episode | ${metrics["dollars_per_episode"]:.4f} |
| cost per hour of episode | ${metrics["dollars_per_video_hour"]:.2f} |

## Files

- `data/test-*.parquet`: the table above, grids included.
- `evalset/droid_evalset_v1.jsonl`: the frozen eval set, as read by `teleop-qc judge`.
- `verdicts/{model}.jsonl`: the judge's raw output, one line per episode.
- `judge/system_prompt.txt`, `judge/answer_schema.json`: what the judge is told and the JSON schema its answer
  must follow.

## License and citation

Apache-2.0, like DROID. The frames come from DROID:

```bibtex
@article{{khazatsky2024droid,
  title   = {{DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset}},
  author  = {{Khazatsky, Alexander and Pertsch, Karl and others}},
  journal = {{arXiv preprint arXiv:2403.12945}},
  year    = {{2024}}
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evalset", type=Path, default=ROOT / "evals/droid_evalset_v1.jsonl")
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--grids", type=Path, required=True, help="directory of <episode>.jpg grids")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from datasets import Dataset, Image
    from huggingface_hub import HfApi

    table = pd.DataFrame(rows(args.evalset, args.verdicts, args.grids))
    verdicts = pd.read_json(args.verdicts, lines=True)
    metrics = report(verdicts)
    model = str(verdicts.model.mode().iat[0])

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    dataset = Dataset.from_pandas(table, preserve_index=False).cast_column("grid", Image())
    dataset.push_to_hub(args.repo, split="test", commit_message="Eval set with grids and verdicts")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "evalset").mkdir()
        (folder / "evalset" / args.evalset.name).write_bytes(args.evalset.read_bytes())
        (folder / "verdicts").mkdir()
        (folder / "verdicts" / f"{model}.jsonl").write_bytes(args.verdicts.read_bytes())
        (folder / "judge").mkdir()
        (folder / "judge" / "system_prompt.txt").write_text(SYSTEM + "\n")
        (folder / "judge" / "answer_schema.json").write_text(json.dumps(SCHEMA, indent=2) + "\n")
        (folder / "README.md").write_text(card(table, metrics, model))
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=folder,
                          commit_message="Dataset card, frozen eval set, raw verdicts, judge prompt and schema")
    print(f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
