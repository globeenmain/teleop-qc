# teleop-qc

Quality control for robot teleoperation data, treated as an ML problem.

Robot learning runs on teleoperation data, and a bad episode costs twice: once when it is collected, and again
when a policy learns from it. This repository checks episodes the way a data lab has to before shipping them:

1. **Signal checks**, before any model looks at the data: dropped or repeated timestamps, frozen cameras, dark
   frames, and the time lag between each camera and the robot's joint encoders, estimated by cross-correlation
   (the idea behind camera-IMU time calibration).
2. **VLM judges**: a vision-language model sees a grid of frames from every camera and the operator's instruction,
   and says whether the robot did what was asked, how confident it is, and what went wrong.
3. **Calibration against human labels**: reliability diagrams, ECE, AUROC and agreement with the people who
   collected the data, recalibration (Platt, isotonic), and the thresholds at which episodes can be accepted or
   rejected without review.

The data is [DROID 1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) (95,658 episodes, 3 cameras,
Apache-2.0), streamed from the Hugging Face Hub: each 500 MB video file holds about 300 episodes, and the code
reads only the byte ranges of the frames it needs. DROID records, for every episode, whether the person who
collected it judged it a success: that is the human label the judges are calibrated against.

## Status

Work in progress, built in the open.

- [x] Streaming access to DROID episodes, frames (AV1, decoded with PyAV) and joint states
- [x] Signal checks, each one tested by injecting the fault it must find
- [x] Calibration toolkit: ECE, AUROC, kappa, Platt and isotonic recalibration, selective thresholds, bootstrap
- [ ] VLM judges (frontier and open-weight), with self-checked structured outputs
- [ ] Eval set and judge outputs published on the Hugging Face Hub
- [ ] Results: accuracy, calibration and cost per processed hour of video, per judge

## Quick start

```bash
pip install -e ".[dev]"
pytest                 # offline tests
pytest -m network      # also reads real DROID episodes from the Hub
```

## How this repository is built

With a coding agent (Claude Code), under one rule: an agent's work is not done until it is verified. Every check
in `signals.py` has a test that plants the fault it must find, every metric in `calibration.py` is tested
against scikit-learn or a case with a known answer, and judges will be gated by evals on a frozen set of
episodes.

## License

Code: Apache-2.0. Data: DROID, Apache-2.0, by the DROID authors.
