# Working on teleop-qc

- Work is done when it is verified: `ruff check .` and `pytest` pass, and `pytest -m network` too when `droid.py`
  changes.
- Every number in the README comes from a run whose output is committed under `results/`, with the command that
  produced it. No estimate is presented as a result.
- A judge never sees the episode's instruction: in DROID 1.0.1 its presence gives the label away (94.6% of
  successful episodes have one, 0.3% of failed ones).
- Judges are scored on `evals/droid_evalset_v1.jsonl`. Changing that set means a new version file, never an edit.
- Before a paid run of more than 10 episodes, run 10, measure the cost, and state the estimate for the full run.
- Credentials: the Claude judge reads `TELEOP_QC_ANTHROPIC_API_KEY`; uploads to the Hugging Face Hub use `HF_TOKEN`
  (account `2skill3d4u`). Never print them.
- Commits are authored by Jonathan Petit <268627786+globeenmain@users.noreply.github.com> and pushed to `main`.
