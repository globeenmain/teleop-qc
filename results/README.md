# Results

Each file here is the output of a command run on the code of this repository; the README quotes nothing else.

| File | Command |
|---|---|
| `signals_droid_file000.csv` | per-stream results of `teleop-qc signals --video-file 0` (main README, first findings) |
| `judge_claude-opus-5-5.jsonl` | `teleop-qc judge --evalset evals/droid_evalset_v1.jsonl --out runs/claude-opus-5-5.jsonl` (default backend `anthropic`, model `claude-opus-5-5`, effort `medium`, 6 moments per camera), run on 2026-11-03 |
| `report_claude-opus-5-5.txt` | first line of the file |
| `reliability_claude-opus-5-5.png` | same command |

## Cost check before the paid run

The judge first ran on the first 10 episodes of the eval set (`--limit 10`, the first 10 lines of
`judge_claude-opus-5-5.jsonl`): 22,660 input and 2,433 output tokens, $0.139 at $4 and $20 per million tokens,
hence an estimate of $2.79 for the 200 episodes. The full run cost $2.92 (`dollars` in the report).

The frame grids the judge saw (one JPEG per episode, 54 MB in all) are not in this repository; they are published
with the eval set on the Hugging Face Hub.
