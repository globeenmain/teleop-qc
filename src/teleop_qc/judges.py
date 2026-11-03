"""VLM judges: did the robot do what the operator was asked to do?

A judge sees one image, a grid of frames from an episode (one column per camera, one row per moment, first frame
at the top), and the instruction, and returns a verdict. The probability of success it implies,
p = confidence if success else 1 - confidence, is what gets calibrated against the human labels.

The judge verifies its own work before a verdict is recorded: the answer must parse, match the schema and be
consistent (a success names no failure mode, a failure names one). An answer that fails the check goes back to
the model once, with the reason; a second failure is recorded as an abstention, never guessed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from PIL import Image

FAILURE_MODES = ("none", "wrong_object", "missed_grasp", "dropped_object", "wrong_placement", "incomplete",
                 "unsafe_motion", "cannot_tell")

SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string", "description": "What the frames show that decides the verdict."},
        "success": {"type": "boolean"},
        "confidence": {"type": "number", "description": "Probability, from 0.5 to 1, that the verdict is right."},
        "failure_mode": {"type": "string", "enum": list(FAILURE_MODES)},
    },
    "required": ["evidence", "success", "confidence", "failure_mode"],
    "additionalProperties": False,
}

SYSTEM = """You review robot teleoperation episodes before they go into a training dataset.

In each episode a person teleoperates a robot arm to do one manipulation task: pick something up and put it \
somewhere, open or close something, press, wipe, pour. Each image is a grid of frames from one episode. Columns \
are cameras: exterior camera 1 on the left, exterior camera 2 in the middle, the wrist camera on the right. Rows \
are moments evenly spaced in time, from the first frame (top row) to the last frame (bottom row).

Decide whether, by the bottom row, the robot has completed the task it was attempting: the object it went for, \
ending in a deliberate place or state. When no instruction is given, infer the task from what the robot \
attempts. Judge only what the frames show. If they do not show enough to tell, answer success false with \
failure_mode "cannot_tell" and a confidence near 0.5.

confidence is your probability that your verdict is right: 0.5 is a coin flip, 1.0 is certainty. Use the whole \
range; a well-calibrated reviewer who says 0.8 is right 8 times out of 10.

Answer with a JSON object with the fields evidence (a few sentences), success, confidence and failure_mode \
(one of: """ + ", ".join(FAILURE_MODES) + ")."

# Models that accept the server-side refusal fallback (`fallbacks: "default"`), from the Claude API docs.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def tile(frames_by_camera: dict[str, np.ndarray], gap: int = 4) -> Image.Image:
    """Grid image with one column per camera and one row per sampled moment (top = first frame)."""
    columns = list(frames_by_camera.values())
    rows, h, w, _ = columns[0].shape
    grid = Image.new("RGB", (len(columns) * w + (len(columns) - 1) * gap, rows * h + (rows - 1) * gap))
    for j, frames in enumerate(columns):
        for i, frame in enumerate(frames):
            grid.paste(Image.fromarray(frame), (j * (w + gap), i * (h + gap)))
    return grid


def jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def user_text(task: str, feedback: str | None) -> str:
    text = f'Instruction given to the operator: "{task}"' if task else "No instruction is available for this episode."
    if feedback:
        text += f"\n\nYour previous answer was rejected: {feedback}. Answer again."
    return text


@dataclass(frozen=True)
class Reply:
    text: str
    model: str  # the model that answered, which a fallback can change
    input_tokens: int
    output_tokens: int


class Refused(Exception):
    """The model declined to answer."""


class Backend(Protocol):
    name: str

    def ask(self, image_jpeg: bytes, task: str, feedback: str | None) -> Reply: ...


@dataclass(frozen=True)
class Verdict:
    success: bool | None  # None: the judge abstained
    confidence: float | None
    p_success: float | None
    failure_mode: str
    evidence: str
    backend: str
    model: str
    attempts: int
    input_tokens: int
    output_tokens: int
    seconds: float
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def check(answer: dict) -> str | None:
    """None when the answer is a valid, consistent verdict; otherwise what is wrong with it."""
    missing = [k for k in SCHEMA["required"] if k not in answer]
    if missing:
        return f"missing fields {missing}"
    confidence = answer["confidence"]
    if not isinstance(answer["success"], bool):
        return "success must be true or false"
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.5 <= confidence <= 1:
        return "confidence must be a number from 0.5 to 1"
    if answer["failure_mode"] not in FAILURE_MODES:
        return f"failure_mode must be one of {', '.join(FAILURE_MODES)}"
    if answer["success"] and answer["failure_mode"] != "none":
        return 'a success has failure_mode "none"'
    if not answer["success"] and answer["failure_mode"] == "none":
        return "a failure must name its failure_mode"
    if not str(answer["evidence"]).strip():
        return "evidence is empty"
    return None


def parse(text: str) -> dict:
    """The JSON object in a reply, tolerating code fences or prose around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def judge(backend: Backend, image_jpeg: bytes, task: str, max_attempts: int = 2) -> Verdict:
    started = time.perf_counter()
    tokens_in = tokens_out = 0
    feedback, model = None, ""
    for attempt in range(1, max_attempts + 1):
        try:
            reply = backend.ask(image_jpeg, task, feedback)
        except Refused as declined:
            feedback = f"refused: {declined}"
            break
        tokens_in, tokens_out, model = tokens_in + reply.input_tokens, tokens_out + reply.output_tokens, reply.model
        try:
            answer = parse(reply.text)
            problem = check(answer) if isinstance(answer, dict) else "the answer is not a JSON object"
        except json.JSONDecodeError as bad:
            problem = f"not valid JSON ({bad.msg})"
        if problem is None:
            confidence = float(answer["confidence"])
            return Verdict(answer["success"], confidence, confidence if answer["success"] else 1 - confidence,
                           answer["failure_mode"], str(answer["evidence"]).strip(), backend.name, model, attempt,
                           tokens_in, tokens_out, time.perf_counter() - started)
        feedback = problem
    return Verdict(None, None, None, "abstained", "", backend.name, model, attempt, tokens_in, tokens_out,
                   time.perf_counter() - started, error=feedback or "")


class AnthropicBackend:
    """A Claude model through the Anthropic API, with structured outputs and the server-side refusal fallback.

    The key is read from TELEOP_QC_ANTHROPIC_API_KEY and the endpoint is set explicitly, so that a coding agent's
    own ANTHROPIC_* variables never leak into the judge.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000,
                 api_key: str | None = None, base_url: str | None = None):
        import anthropic

        key = api_key or os.environ.get("TELEOP_QC_ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("set TELEOP_QC_ANTHROPIC_API_KEY to run the Claude judge")
        self.client = anthropic.Anthropic(
            api_key=key, base_url=base_url or os.environ.get("TELEOP_QC_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            max_retries=3, timeout=180.0)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def ask(self, image_jpeg: bytes, task: str, feedback: str | None) -> Reply:
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} \
            if self.model in _FALLBACK_MODELS else {}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(image_jpeg).decode()}},
                {"type": "text", "text": user_text(task, feedback)},
            ]}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            **extra,
        )
        if response.stop_reason == "refusal":
            details = response.stop_details
            raise Refused(getattr(details, "category", None) or "no category")
        text = next((b.text for b in response.content if b.type == "text"), "")
        if response.stop_reason == "max_tokens":
            text = ""  # cut off: the check rejects it and the judge asks again
        return Reply(text, response.model, response.usage.input_tokens, response.usage.output_tokens)


class HuggingFaceBackend:
    """An open-weight VLM through Hugging Face Inference Providers (token in HF_TOKEN)."""

    name = "huggingface"

    def __init__(self, model: str = "Qwen/Qwen2.5-VL-7B-Instruct", provider: str = "auto",
                 token: str | None = None, max_tokens: int = 1024):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(provider=provider, api_key=token or os.environ.get("HF_TOKEN"))
        self.model, self.max_tokens = model, max_tokens

    def ask(self, image_jpeg: bytes, task: str, feedback: str | None) -> Reply:
        url = "data:image/jpeg;base64," + base64.standard_b64encode(image_jpeg).decode()
        out = self.client.chat_completion(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": user_text(task, feedback)}]},
            ],
        )
        usage = out.usage
        return Reply(out.choices[0].message.content or "", self.model,
                     getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)
