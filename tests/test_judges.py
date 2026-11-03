"""The judge loop with scripted backends: valid answers, self-correction, abstention."""

import json

import numpy as np
import pytest

from teleop_qc import judges


class Scripted:
    """A backend that returns prepared replies in order and records what it was asked."""

    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.feedback = []

    def ask(self, image_jpeg, task, feedback):
        self.feedback.append(feedback)
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return judges.Reply(text, "scripted-1", input_tokens=1000, output_tokens=50)


def answer(**fields):
    base = {"evidence": "The marker ends inside the cup.", "success": True, "confidence": 0.9,
            "failure_mode": "none"}
    return json.dumps(base | fields)


def test_valid_answer_gives_a_probability_of_success():
    verdict = judges.judge(Scripted(answer()), b"jpeg", "Put the marker in the cup")
    assert verdict.success and verdict.p_success == pytest.approx(0.9) and verdict.attempts == 1
    failed = judges.judge(Scripted(answer(success=False, confidence=0.7, failure_mode="missed_grasp")), b"", "t")
    assert failed.p_success == pytest.approx(0.3)


def test_inconsistent_answer_goes_back_once_with_the_reason():
    backend = Scripted(answer(success=True, failure_mode="dropped_object"), answer())
    verdict = judges.judge(backend, b"jpeg", "task")
    assert verdict.success and verdict.attempts == 2
    assert backend.feedback == [None, 'a success has failure_mode "none"']
    assert verdict.input_tokens == 2000  # both attempts are paid for


def test_two_bad_answers_make_an_abstention_not_a_guess():
    verdict = judges.judge(Scripted("not json at all", answer(confidence=1.4)), b"", "task")
    assert verdict.success is None and verdict.p_success is None
    assert "confidence" in verdict.error


def test_refusal_is_an_abstention():
    verdict = judges.judge(Scripted(judges.Refused("cyber")), b"", "task")
    assert verdict.success is None and "refused" in verdict.error


@pytest.mark.parametrize("bad, reason", [
    ({"success": True, "confidence": 0.9, "failure_mode": "none"}, "missing"),
    ({"evidence": "x", "success": "yes", "confidence": 0.9, "failure_mode": "none"}, "true or false"),
    ({"evidence": "x", "success": True, "confidence": 0.2, "failure_mode": "none"}, "0.5 to 1"),
    ({"evidence": "x", "success": False, "confidence": 0.8, "failure_mode": "none"}, "name its failure_mode"),
    ({"evidence": "x", "success": False, "confidence": 0.8, "failure_mode": "gremlins"}, "one of"),
    ({"evidence": " ", "success": True, "confidence": 0.8, "failure_mode": "none"}, "empty"),
])
def test_check_explains_what_is_wrong(bad, reason):
    assert reason in judges.check(bad)


def test_parse_tolerates_code_fences():
    assert judges.parse('```json\n{"a": 1}\n```') == {"a": 1}


def test_tile_puts_cameras_in_columns_and_time_in_rows():
    frames = {cam: np.full((6, 180, 320, 3), i * 100, dtype=np.uint8) for i, cam in enumerate("abc")}
    grid = judges.tile(frames, gap=4)
    assert grid.size == (3 * 320 + 2 * 4, 6 * 180 + 5 * 4)
    assert grid.getpixel((330, 10)) == (100, 100, 100)  # second column = second camera
    assert len(judges.jpeg(grid)) > 0
