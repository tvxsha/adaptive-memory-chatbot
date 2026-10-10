"""Offline tests for the evaluation helpers (no network, no LLM)."""
from evaluation import locomo_utils as lu


def _entry():
    return {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Mel",
            # session_2 is listed before session_10 on purpose: sorting must be numeric
            "session_10": [{"speaker": "Mel", "dia_id": "D10:1", "text": "Ten"}],
            "session_10_date_time": "9:00 am on 1 Dec, 2023",
            "session_2": [
                {"speaker": "Caroline", "dia_id": "D2:1", "text": "  Two  "},
                {"speaker": "Mel", "dia_id": "D2:2", "text": ""},
                {"speaker": "Mel", "dia_id": "D2:3", "text": "Look!", "blip_caption": "a red kite"},
            ],
            "session_2_date_time": "1:00 pm on 5 May, 2023",
            "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "One"}],
            "session_1_date_time": "8:00 am on 1 May, 2023",
        },
        "qa": [
            {"question": "q0", "answer": "a0", "evidence": ["D1:1"], "category": 4},
            {"question": "q1", "answer": 2023, "evidence": ["D2:1; D10:1"], "category": 2},
            {"question": "q2", "evidence": ["D1:1"], "category": 5, "adversarial_answer": "x"},
            {"question": "q3", "answer": "a3", "evidence": [], "category": 1},
        ],
    }


def test_turns_are_chronological_and_numeric_sorted():
    ids = [t["dia_id"] for t in lu.conversation_turns(_entry())]
    assert ids == ["D1:1", "D2:1", "D2:3", "D10:1"]  # empty D2:2 dropped


def test_text_is_stripped_and_caption_appended():
    turns = {t["dia_id"]: t for t in lu.conversation_turns(_entry())}
    assert turns["D2:1"]["text"] == "Two"
    assert turns["D2:3"]["text"] == "Look! (shared a photo: a red kite)"


def test_captions_can_be_turned_off():
    turns = {t["dia_id"]: t for t in lu.conversation_turns(_entry(), include_captions=False)}
    assert turns["D2:3"]["text"] == "Look!"


def test_format_turn_contains_date_and_speaker():
    turn = lu.conversation_turns(_entry())[0]
    assert lu.format_turn(turn) == "[8:00 am on 1 May, 2023] Caroline: One"


def test_evidence_ids_handles_malformed_entries():
    assert lu.evidence_ids(["D8:6; D9:17", "D1:2", "garbage", None]) == ["D8:6", "D9:17", "D1:2"]
    assert lu.evidence_ids(None) == []


def test_answerable_questions_drop_adversarial_and_keep_index():
    qs = lu.answerable_questions(_entry())
    assert [q["idx"] for q in qs] == [0, 1, 3]
    assert qs[1]["answer"] == "2023"  # non-string answers become strings
    assert qs[1]["evidence"] == ["D2:1", "D10:1"]


def test_token_f1():
    assert lu.token_f1("7 May 2023", "7 May 2023") == 1.0
    assert lu.token_f1("The adoption agencies", "adoption agencies") == 1.0  # articles ignored
    assert lu.token_f1("pottery", "painting") == 0.0
    assert 0 < lu.token_f1("adoption agencies in Texas", "adoption agencies") < 1
    assert lu.token_f1("", "") == 1.0
    assert lu.token_f1("", "something") == 0.0


def test_approx_tokens():
    assert lu.approx_tokens("one two three four five") == 6


def test_evidence_hit_all_vs_any():
    ev = ["A", "B"]
    assert lu.evidence_hit(ev, ["A", "C"], require_all=False) is True
    assert lu.evidence_hit(ev, ["A", "C"], require_all=True) is False
    assert lu.evidence_hit(ev, ["B", "A"], require_all=True) is True
    assert lu.evidence_hit([], ["A"], require_all=True) is False  # no evidence -> not a hit


def test_evidence_texts_skips_unknown_ids():
    turns = lu.conversation_turns(_entry())
    by_id = {t["dia_id"]: t for t in turns}
    assert lu.evidence_texts(by_id, ["D1:1", "D99:9"]) == [lu.format_turn(by_id["D1:1"])]

def test_questions_within_keeps_only_fully_covered_questions():
    turns = [{"dia_id": "D1:1"}, {"dia_id": "D1:2"}]
    questions = [
        {"idx": 0, "evidence": ["D1:1"]},
        {"idx": 1, "evidence": ["D1:1", "D2:5"]},  # one evidence turn is in the future
        {"idx": 2, "evidence": []},                # no evidence -> cannot be checked
        {"idx": 3, "evidence": ["D1:2"]},
    ]
    assert [q["idx"] for q in lu.questions_within(questions, turns)] == [0, 3]