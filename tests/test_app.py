"""Headless UI tests: every button is clicked through Streamlit's AppTest runner and the
values on screen are compared with the algorithm and the export functions."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from dna_alignment import EXAMPLES, needleman_wunsch, smith_waterman  # noqa: E402
from dna_alignment.export import stats_for  # noqa: E402

from pathlib import Path  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def fresh():
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def button(at, label, key=None):
    for b in at.button:
        if b.label == label and (key is None or b.key == key):
            return b
    raise AssertionError(f"button {label!r} not found")


def test_first_load_shows_worked_result():
    at = fresh()
    r = needleman_wunsch("AGTCA", "AGCA")
    assert metric(at, "Optimal score") == str(r.score)
    assert metric(at, "Identity") == f"{stats_for(r).identity:.2f}%"


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda e: e.key)
def test_each_example_loads_and_matches_algorithm(ex):
    at = fresh()
    at.selectbox(key="example_key").set_value(ex.key).run()
    button(at, "Load example").click().run()
    assert not at.exception
    st_ = stats_for(needleman_wunsch(ex.seq1, ex.seq2))
    assert metric(at, "Optimal score") == str(st_.score)
    assert metric(at, "Matches") == str(st_.matches)
    assert metric(at, "Mismatches") == str(st_.mismatches)
    assert metric(at, "Gap positions") == str(st_.gaps)
    assert metric(at, "Alignment length") == str(st_.length)


def test_validation_lowercase_invalid_and_clear():
    at = fresh()
    at.text_area(key="seq1_in").input("ag tc").run()
    assert any("length **4**" in c.value for c in at.caption)
    at.text_area(key="seq2_in").input("AGXT").run()
    assert any("invalid character" in e.value for e in at.error)
    assert button(at, "▶ Run alignment").disabled
    button(at, "Clear").click().run()
    assert at.text_area(key="seq1_in").value == ""
    assert any("Run alignment" in i.value for i in at.info)


def test_custom_scoring_and_validation():
    at = fresh()
    at.number_input(key="match_in").set_value(1).run()
    at.number_input(key="mismatch_in").set_value(1).run()
    assert any("greater than the mismatch" in e.value for e in at.error)
    at.number_input(key="mismatch_in").set_value(-3).run()
    at.number_input(key="gap_in").set_value(-1).run()
    button(at, "▶ Run alignment").click().run()
    from dna_alignment import ScoringScheme
    r = needleman_wunsch("AGTCA", "AGCA", ScoringScheme(1, -3, -1))
    assert metric(at, "Optimal score") == str(r.score)
    button(at, "Restore defaults (+2 / −1 / −2)").click().run()
    assert at.number_input(key="gap_in").value == -2


def test_random_generators():
    at = fresh()
    at.number_input(key="len1").set_value(12).run()
    at.number_input(key="len2").set_value(9).run()
    button(at, "🎲 Both").click().run()
    s1, s2 = at.text_area(key="seq1_in").value, at.text_area(key="seq2_in").value
    assert len(s1) == 12 and len(s2) == 9 and set(s1 + s2) <= set("ACGT")
    assert metric(at, "Optimal score") == str(needleman_wunsch(s1, s2).score)
    button(at, "🎲 Seq 1").click().run()
    assert at.text_area(key="seq2_in").value == s2
    button(at, "🎲 Seq 2").click().run()
    assert not at.exception


def test_step_controls():
    at = fresh()
    K = len(needleman_wunsch("AGTCA", "AGCA").fill_steps)
    button(at, "Next ▶").click().run()
    assert at.session_state.fill_step == 1
    button(at, "Next ▶").click().run()
    assert at.session_state.fill_step == 2
    button(at, "◀ Prev").click().run()
    assert at.session_state.fill_step == 1
    at.slider(key="fill_step").set_value(K).run()
    assert any(f"Step {K} of {K}" in md.value for md in at.markdown)
    button(at, "↺ Reset").click().run()
    assert at.session_state.fill_step == 0
    button(at, "⏮ Start").click().run()
    assert at.session_state.fill_step == 1
    button(at, "⏵ Play").click().run()
    assert at.session_state.fill_playing and not at.exception


def test_traceback_controls():
    at = fresh()
    T = len(needleman_wunsch("AGTCA", "AGCA").traceback_steps)
    button(at, "Next ▶", key="tb_next").click().run()
    assert at.session_state.tb_step == 1
    at.slider(key="tb_step").set_value(T - 1).run()
    assert any(f"Traceback step {T} of {T}" in md.value for md in at.markdown)
    button(at, "↺ Reset", key="tb_reset").click().run()
    assert at.session_state.tb_step == 0


def test_local_mode_and_comparison():
    at = fresh()
    at.text_area(key="seq1_in").input("TTTTGATTACATTTT").run()
    at.text_area(key="seq2_in").input("CCGATTACACC").run()
    at.radio(key="algo_in").set_value("Smith-Waterman").run()
    button(at, "▶ Run alignment").click().run()
    assert metric(at, "Optimal score") == str(smith_waterman("TTTTGATTACATTTT", "CCGATTACACC").score)
    assert metric(at, "Score") is not None and not at.exception


def test_empty_and_large_inputs_do_not_crash():
    at = fresh()
    at.text_area(key="seq1_in").input("").run()
    button(at, "▶ Run alignment").click().run()
    assert not at.exception
    at.number_input(key="len1").set_value(300).run()
    at.number_input(key="len2").set_value(300).run()
    button(at, "🎲 Both").click().run()
    assert not at.exception
    assert any("too many to draw" in i.value for i in at.info)
