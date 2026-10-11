"""End-to-end tests that drive the real Streamlit pages. Gemma is faked at the network layer, so no Ollama is needed.

Run with:  python -m pytest -q
"""
import datetime as dt
import json

import pytest
import requests
from streamlit.testing.v1 import AppTest

import logic as g

APP = str(__import__("pathlib").Path(__file__).with_name("app.py"))
NEW = "+ New plant"


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": json.dumps(self._content)}}


def fake_gemma(url, json=None, timeout=None):
    system, user = json["messages"][0]["content"], json["messages"][1]["content"]
    if system.startswith("You turn a gardener's note"):
        return FakeResponse({"noticed": [user.split("Note: ", 1)[1]], "conditions": [], "comparison": ""})
    if system.startswith("You help a gardener observe"):
        return FakeResponse({"decision": "LOOK_FIRST", "question": "Are the dark buds getting longer?",
                             "reason": "Look before changing anything."})
    return FakeResponse({"choice": "brighter", "measure": "open flowers"})


@pytest.fixture
def garden(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "BASE_DIR", tmp_path)
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "garden.json")
    monkeypatch.setattr(g, "PHOTO_DIR", tmp_path / "photos")
    monkeypatch.setattr(requests, "post", fake_gemma)
    return tmp_path


def new_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"no button {label!r}; have {[b.label for b in at.button]}")


def saved(garden):
    return json.loads((garden / "garden.json").read_text(encoding="utf-8"))


def record_note(at, plant, text, day):
    at.selectbox(key="obs_plant").set_value(NEW if plant not in at.selectbox(key="obs_plant").options else plant)
    at.run()
    if at.selectbox(key="obs_plant").value == NEW:
        at.text_input(key="new_plant").set_value(plant)
    at.text_area(key="obs_note").set_value(text)
    at.date_input(key="obs_date").set_value(day)
    button(at, "Record observation").click()
    at.run()
    assert not at.exception, at.exception


def test_first_screen_is_calm_when_nothing_is_due(garden):
    at = new_app()
    assert any("Nothing is due" in c.value for c in at.caption)
    assert not (garden / "garden.json").exists()


def test_new_plant_flow_keeps_the_plant_selected_and_clears_the_note(garden):
    at = new_app()
    record_note(at, "Petunia", "2 open flowers, 4 buds", dt.date(2026, 10, 7))
    assert at.selectbox(key="obs_plant").value == "Petunia"
    assert at.text_area(key="obs_note").value == ""
    assert any(s.value == "Saved." for s in at.success)
    assert any("LOOK FIRST" in w.value for w in at.warning)
    entry = saved(garden)["plants"]["Petunia"]["observations"][0]
    assert entry["date"] == "2026-10-07" and entry["logged_on"] == g.today()
    at.run()  # any later interaction must not reset the picker to another plant
    assert at.selectbox(key="obs_plant").value == "Petunia"
    assert any("Are the dark buds getting longer?" in c.value for c in at.markdown)
    assert any("Go and look" in m.value for m in at.markdown)


def test_full_loop_look_first_then_watch_with_backdated_counts(garden):
    at = new_app()
    record_note(at, "Petunia", "2 open flowers, 4 buds", dt.date(2026, 10, 7))
    at.text_area(key="found|Petunia").set_value("Buds are a little longer than yesterday.")
    button(at, "Save what I found").click()
    at.run()
    assert not at.exception, at.exception
    assert saved(garden)["plants"]["Petunia"]["ready_for_trial"] is True

    at.selectbox(key="trial_plant").set_value("Petunia")
    at.run()
    at.button(key="watch|Petunia").click()
    at.run()
    assert not at.exception, at.exception

    for n, (day, flowers, buds) in enumerate([(dt.date(2026, 10, 7), 2, 4), (dt.date(2026, 10, 10), 4, 2)]):
        at.date_input(key=f"date|Petunia|0|{n}").set_value(day)
        at.number_input(key=f"n1|Petunia|0|{n}").set_value(flowers)
        at.number_input(key=f"n2|Petunia|0|{n}").set_value(buds)
        at.button(key="add|Petunia|0").click()
        at.run()
        assert not at.exception, at.exception

    texts = " ".join(m.value for m in at.info) + " ".join(m.value for m in at.markdown)
    assert "Observation only" in texts and "says nothing about a cause" in texts
    assert "Evidence level 0 of 3" in texts and "Next step" in texts
    assert len(at.get("vega_lite_chart")) >= 1
    records = saved(garden)["plants"]["Petunia"]["trials"][0]["records"]
    assert [r["date"] for r in records] == ["2026-10-07", "2026-10-10"]


def test_count_check_blocks_a_big_jump_until_confirmed(garden):
    at = new_app()
    record_note(at, "Rose", "1 bud and 2 leaves", dt.date(2026, 10, 7))
    at.text_area(key="found|Rose").set_value("The bud is green with a pink tip.")
    button(at, "Save what I found").click()
    at.run()
    at.selectbox(key="trial_plant").set_value("Rose")
    at.run()
    at.button(key="watch|Rose").click()
    at.run()
    at.date_input(key="date|Rose|0|0").set_value(dt.date(2026, 10, 7))
    at.number_input(key="n1|Rose|0|0").set_value(2)
    at.button(key="add|Rose|0").click()
    at.run()
    at.number_input(key="n1|Rose|0|1").set_value(12)
    at.date_input(key="date|Rose|0|1").set_value(dt.date(2026, 10, 10))
    at.run()
    assert any("Big change" in w.value for w in at.warning)
    assert at.button(key="add|Rose|0").disabled
    at.checkbox(key="jumpok|Rose|0|1").check()
    at.run()
    assert not at.button(key="add|Rose|0").disabled


def test_ollama_down_shows_an_error_and_creates_nothing(garden, monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", down)
    at = new_app()
    record_note(at, "Petunia", "2 open flowers", dt.date(2026, 10, 7))
    assert any("Cannot reach Ollama" in e.value for e in at.error)
    assert not (garden / "garden.json").exists()
    assert "Petunia" not in [o for o in at.selectbox(key="obs_plant").options]


def test_blank_note_is_refused(garden):
    at = new_app()
    at.selectbox(key="obs_plant").set_value(NEW)
    at.run()
    at.text_input(key="new_plant").set_value("Petunia")
    button(at, "Record observation").click()
    at.run()
    assert any("Add a plant name and a note" in e.value for e in at.error)
    assert not (garden / "garden.json").exists()


def test_unreadable_garden_file_is_kept_and_the_user_is_told(garden):
    (garden / "garden.json").write_text("{broken", encoding="utf-8")
    at = new_app()
    assert any("could not be read" in w.value for w in at.warning)
    assert len(list(garden.glob("garden.unreadable-*.json"))) == 1


def test_sample_garden_works_and_never_touches_the_real_file(garden):
    at = new_app()
    at.checkbox(key="sample_mode").check()
    at.run()
    assert not at.exception, at.exception
    assert any("SAMPLE GARDEN" in w.value for w in at.warning)
    assert any("Go and look" in m.value for m in at.markdown)  # the sample mint question is due
    assert any(b.label.startswith("Repeat this trial") for b in at.button)
    record_note(at, "Sample basil (illustrative)", "3 new leaves today", dt.date(2026, 10, 9))
    assert not (garden / "garden.json").exists()
    assert not (garden / "photos").exists()


def test_switching_between_sample_and_real_never_crashes(garden):
    at = new_app()
    record_note(at, "Petunia", "2 open flowers", dt.date(2026, 10, 7))
    at.checkbox(key="sample_mode").check()
    at.run()
    assert not at.exception, at.exception
    at.checkbox(key="sample_mode").uncheck()
    at.run()
    assert not at.exception, at.exception
    assert "Petunia" in at.selectbox(key="obs_plant").options


def test_plant_details_are_saved_and_never_sent_to_gemma(garden, monkeypatch):
    seen = []

    def spy(url, json=None, timeout=None):
        seen.append(json["messages"][1]["content"])
        return fake_gemma(url, json=json, timeout=timeout)

    monkeypatch.setattr(requests, "post", spy)
    at = new_app()
    record_note(at, "Bougainvillea", "1 of 5 cuttings has 2 pink bracts", dt.date(2026, 10, 10))
    at.text_area(key="facts|Bougainvillea").set_value("Mother plant is in full bloom. Different varieties.")
    button(at, "Save details").click()
    at.run()
    assert saved(garden)["plants"]["Bougainvillea"]["facts"].startswith("Mother plant is in full bloom")
    assert seen and not any("Mother plant" in s for s in seen)


def test_last_photo_is_shown_beside_the_uploader_and_in_history(garden):
    from PIL import Image

    (garden / "photos").mkdir()
    Image.new("RGB", (400, 300), "orange").save(garden / "photos" / "marigold-2026-10-05-1.jpg")
    plant = g.get_plant({"plants": {}}, "Marigold")
    data = {"plants": {"Marigold": plant}}
    plant["observations"] = [{"date": "2026-10-05", "kind": "observation", "text": "3 open flowers"}]
    plant["ready_for_trial"] = True
    plant["trials"] = [{"question": "q", "hypothesis": "h", "change": "Nothing (observe only)", "keep_same": "x",
                        "comparison": "none", "measure": "open flowers", "measure2": "", "duration_days": 3,
                        "start": "2026-10-05", "status": "active", "has_control": False, "observe_only": True,
                        "records": [{"date": "2026-10-05", "changed": 3, "control": 0, "logged_on": "2026-10-09",
                                     "photo": "photos/marigold-2026-10-05-1.jpg"}]}]
    g.save_data(data)
    at = new_app()
    at.selectbox(key="trial_plant").set_value("Marigold")
    at.run()
    assert not at.exception, at.exception
    captions = [img.proto.imgs[0].caption for img in at.get("image")]
    assert len(captions) == 3  # preview beside the uploader, photo under the table, photo in History
    assert captions[0].startswith("Last photo (2026-10-05)")
    assert captions[1:] == ["2026-10-05 · open flowers: 3"] * 2
    assert any("Day " in m.value and "count open flowers" in m.value for m in at.markdown)  # Today panel


def test_without_gemma_the_whole_loop_works_with_no_network_at_all(garden, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("the app must not call Ollama when Gemma is switched off")

    monkeypatch.setattr(requests, "post", no_network)
    at = new_app()
    at.checkbox(key="no_gemma").check()
    at.run()
    record_note(at, "Petunia", "2 open flowers, 4 buds", dt.date(2026, 10, 7))
    assert any("kept exactly as you wrote it" in m.value for m in at.markdown)
    assert any("Gemma is switched off" in c.value for c in at.caption)
    stored = saved(garden)
    assert stored["plants"]["Petunia"]["observations"][0]["structured"]["parsed_by"] == "no_model"
    assert stored["plants"]["Petunia"]["pending"]["question"] == g.FALLBACK_QUESTION
    assert "guard_events" not in stored  # nothing the model did wrong is claimed

    at.text_area(key="found|Petunia").set_value("Two buds are bigger and darker today.")
    button(at, "Save what I found").click()
    at.run()
    at.selectbox(key="trial_plant").set_value("Petunia")
    at.run()
    assert button(at, "Suggest a trial").disabled
    button(at, "Use this change").click()
    at.run()
    button(at, "Start this trial").click()
    at.run()
    assert not at.exception, at.exception
    assert saved(garden)["plants"]["Petunia"]["trials"][0]["status"] == "active"


def test_the_ollama_error_points_to_the_no_gemma_switch(garden, monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", down)
    at = new_app()
    record_note(at, "Petunia", "2 open flowers", dt.date(2026, 10, 7))
    assert any("Run without Gemma" in e.value for e in at.error)


def test_sidebar_does_not_claim_gemma_reads_notes_when_it_is_off(garden):
    claim = "Notes are processed by a local open-weight Gemma model"
    at = new_app()
    assert any(claim in c.value for c in at.caption)
    at.checkbox(key="no_gemma").check()
    at.run()
    assert not any(claim in c.value for c in at.caption)
    assert any("Gemma is off" in c.value for c in at.caption)


def test_look_first_label_says_when_the_pest_guard_replaced_gemmas_question(garden):
    data = {"plants": {"rose": {"observations": [{"date": "2026-10-10", "kind": "observation", "text": "a bud"}],
                                "trials": [], "ready_for_trial": False,
                                "pending": {"decision": "LOOK_FIRST", "question": "Count the buds.",
                                            "reason": "Look before changing anything.", "source": "guard"}}}}
    (garden / "garden.json").write_text(json.dumps(data), encoding="utf-8")
    at = new_app()
    labels = [c.value for c in at.caption if c.value.startswith("Question written by")]
    assert labels == ["Question written by: the app's built-in question "
                      "(Gemma's question was about pests or disease, which your notes did not mention)"]