"""Command line: build an eval set, run the signal checks, run a judge, report on its verdicts.

    teleop-qc evalset --n 200 --out runs/evalset.jsonl
    teleop-qc signals --video-file 0 --out runs/signals.parquet
    teleop-qc judge   --evalset runs/evalset.jsonl --backend anthropic --out runs/claude-opus-5-5.jsonl
    teleop-qc report  runs/claude-opus-5-5.jsonl --evalset runs/evalset.jsonl --plot runs/reliability.png
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as cal
from .droid import CAMERAS, Episode, VideoFile, VideoSpan, load_episodes, sample_balanced
from .judges import AnthropicBackend, HuggingFaceBackend, jpeg, judge, tile
from .pipeline import run_signals, signals_summary

# $ per million input and output tokens, Claude API list prices (November 2026)
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def save_evalset(episodes: list[Episode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in episodes:
            f.write(json.dumps(asdict(e)) + "\n")


def load_evalset(path: Path) -> list[Episode]:
    episodes = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row["videos"] = tuple(VideoSpan(**v) for v in row["videos"])
        episodes.append(Episode(**row))
    return episodes


class FrameSource:
    """Keeps the last few video files open, since consecutive episodes usually live in the same files."""

    def __init__(self, keep: int = 6):
        self.open: OrderedDict[str, VideoFile] = OrderedDict()
        self.keep = keep

    def frames(self, span: VideoSpan, k: int) -> np.ndarray:
        if span.path not in self.open:
            if len(self.open) >= self.keep:
                self.open.popitem(last=False)[1].close()
            self.open[span.path] = VideoFile(span.path)
        self.open.move_to_end(span.path)
        return self.open[span.path].frames_at(span.timestamps(k))


def cmd_evalset(args) -> None:
    episodes = [e for f in args.meta_files for e in load_episodes(f)]
    # In DROID 1.0.1 almost no failed episode has an instruction, so requiring one would leave no failures, and
    # showing it would give the label away: the judge never sees the instruction, for either class.
    sample = sample_balanced(episodes, args.n, seed=args.seed, require_task=False)
    save_evalset(sample, args.out)
    print(f"{len(sample)} episodes ({sum(e.success for e in sample)} successes) -> {args.out}")


def cmd_signals(args) -> None:
    if args.evalset:
        episodes = load_evalset(args.evalset)
    else:  # every episode stored in one video file (of the first exterior camera)
        episodes = [e for e in load_episodes(0) if e.video(CAMERAS[0]).file == args.video_file
                    and e.video(CAMERAS[0]).chunk == 0]
    df, offsets = run_signals(episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(signals_summary(df, offsets).round(3).to_string(index=False))


def cmd_judge(args) -> None:
    if args.backend == "anthropic":
        backend = AnthropicBackend(model=args.model or "claude-opus-5-5", effort=args.effort)
    else:
        backend = HuggingFaceBackend(model=args.model or "Qwen/Qwen2.5-VL-7B-Instruct")
    episodes = load_evalset(args.evalset)[: args.limit]
    done = {json.loads(line)["episode_index"] for line in args.out.read_text().splitlines()} \
        if args.out.exists() else set()
    images = args.out.with_suffix("")  # grids kept next to the verdicts, for audit and for the eval set
    images.mkdir(parents=True, exist_ok=True)
    source = FrameSource()
    with args.out.open("a") as out:
        for n, episode in enumerate(e for e in episodes if e.index not in done):
            grid = tile({cam: source.frames(episode.video(cam), args.frames) for cam in CAMERAS})
            image = jpeg(grid)
            (images / f"{episode.index:06d}.jpg").write_bytes(image)
            verdict = judge(backend, image, task="")  # see cmd_evalset: the instruction would leak the label
            out.write(json.dumps({"episode_index": episode.index, "label": episode.success,
                                  "seconds_of_video": episode.seconds, **verdict.as_dict()}) + "\n")
            out.flush()
            print(f"[{n + 1}] episode {episode.index}: label={episode.success} "
                  f"p={verdict.p_success} ({verdict.failure_mode}) {verdict.seconds:.1f}s")


def report(rows: pd.DataFrame) -> dict:
    """Judge metrics against the human labels, on the episodes where the judge answered."""
    answered = rows[rows.p_success.notna()]
    p, y = answered.p_success.to_numpy(float), answered.label.to_numpy(int)
    out = {"episodes": len(rows), "abstained": int(len(rows) - len(answered)), **cal.summary(p, y)}
    verdict = p >= 0.5  # the same cut as the accuracy in cal.summary
    out["judged_successful"] = float(verdict.mean())
    out["true_positive_rate"] = float(verdict[y == 1].mean())  # human successes the judge calls successes
    out["true_negative_rate"] = float((~verdict)[y == 0].mean())  # human failures the judge calls failures
    out["balanced_accuracy"] = (out["true_positive_rate"] + out["true_negative_rate"]) / 2
    out["auroc_ci"] = cal.bootstrap_ci(cal.auroc, p, y)
    recalibrated = cal.cross_fit(p, y, method="isotonic")
    out["ece_after_isotonic"] = cal.ece(recalibrated, y)
    accept = cal.selective_threshold(recalibrated, y, 0.95, accept=True)
    reject = cal.selective_threshold(recalibrated, y, 0.90, accept=False)
    out["auto_accept_at_95"] = accept.coverage if accept else 0.0
    out["auto_reject_at_90"] = reject.coverage if reject else 0.0
    price = PRICES.get(str(rows.model.mode().iat[0]) if rows.model.notna().any() else "", None)
    if price:
        dollars = (rows.input_tokens.sum() * price[0] + rows.output_tokens.sum() * price[1]) / 1e6
        out["dollars"] = dollars
        out["dollars_per_episode"] = dollars / len(rows)
        out["dollars_per_video_hour"] = dollars / (rows.seconds_of_video.sum() / 3600)
    out["seconds_per_episode"] = float(rows.seconds.mean())
    return out


def plot_reliability(rows: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    answered = rows[rows.p_success.notna()]
    p, y = answered.p_success.to_numpy(float), answered.label.to_numpy(int)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")
    for probs, name in ((p, "raw"), (cal.cross_fit(p, y, method="isotonic"), "isotonic, cross-fitted")):
        bins = cal.reliability_bins(probs, y, n_bins=8, strategy="quantile")
        ax.plot([b.confidence for b in bins], [b.frequency for b in bins], marker="o", label=name)
    ax.set(xlabel="judge's probability of success", ylabel="share labelled successful by humans",
           xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def cmd_report(args) -> None:
    for path in args.verdicts:
        rows = pd.read_json(path, lines=True)
        metrics = report(rows)
        print(f"== {path}")
        for key, value in metrics.items():
            print(f"{key:>24}: {value:.3f}" if isinstance(value, float) else f"{key:>24}: {value}")
        answered = rows[rows.p_success.notna()]
        bins = cal.reliability_bins(answered.p_success.to_numpy(float), answered.label.to_numpy(int), n_bins=8,
                                    strategy="quantile")
        print("\nreliability diagram, raw probabilities (8 quantile bins):")
        print(pd.DataFrame([vars(b) for b in bins]).round(3).to_string(index=False))
        print("\nfailure modes named by the judge, by human label:")
        print(pd.crosstab(rows.failure_mode, rows.label.map({True: "success", False: "failure"})).to_string())
        if args.plot:
            plot_reliability(rows, args.plot, Path(path).stem)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="teleop-qc", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evalset", help="draw a balanced set of episodes with an instruction")
    p.add_argument("--meta-files", type=int, nargs="+", default=[0])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=cmd_evalset)

    p = sub.add_parser("signals", help="timestamp, frame and camera-sync checks")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--evalset", type=Path)
    source.add_argument("--video-file", type=int, help="all episodes of this video file (chunk 0)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=cmd_signals)

    p = sub.add_parser("judge", help="run a VLM judge over the eval set (resumable)")
    p.add_argument("--evalset", type=Path, required=True)
    p.add_argument("--backend", choices=("anthropic", "huggingface"), default="anthropic")
    p.add_argument("--model")
    p.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    p.add_argument("--frames", type=int, default=6, help="moments sampled per camera")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=cmd_judge)

    p = sub.add_parser("report", help="calibration and cost of one or more judges")
    p.add_argument("verdicts", nargs="+", type=Path)
    p.add_argument("--plot", type=Path)
    p.set_defaults(run=cmd_report)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
