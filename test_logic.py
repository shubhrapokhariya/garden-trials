"""Tests for Garden Trials logic. Gemma is replaced by fake output, so no Ollama is needed.

Run with:  python -m pytest -q
"""
import io
import json

import pytest
from PIL import Image

import logic as g


def fake(out):
    return lambda model, system, user: out


def plant_with(*kinds, text=None):
    return {
        "observations": [
            {"date": f"2026-10-0{i + 1}", "kind": k, "text": text or f"note number {i}"}
            for i, k in enumerate(kinds)
        ],
        "pending": None, "ready_for_trial": False, "trials": [],
    }


def trial(records, has_control=True, observe_only=False, status="active", change="Move to more sun",
          start_changes=None):
    return {"records": records, "has_control": has_control, "observe_only": observe_only,
            "status": status, "change": change, "measure": "flowers", "measure2": "",
            "start_changes": start_changes or []}


def rec(date, changed, control=0, **extra):
    return {"date": date, "changed": changed, "control": control, **extra}


# ---------- extract_observation ----------
def test_good_reading_is_kept_and_cause_stays_unknown():
    out = {"noticed": ["3 yellow leaves", "stem looks thin"], "conditions": [], "comparison": "",
           "cause": "overwatering"}
    r = g.extract_observation("m", "basil", "3 yellow leaves, stem looks thin", ask=fake(out))
    assert r["parsed_by"] == "gemma"
    assert r["cause"] == "unknown"
    assert g.events_from_extract(r) == []


def test_unusable_output_keeps_the_gardeners_words():
    r = g.extract_observation("m", "basil", "3 yellow leaves", ask=fake(None))
    assert r["parsed_by"] == "fallback"
    assert r["noticed"] == ["3 yellow leaves"]
    assert g.events_from_extract(r) == ["unusable_output"]


def test_dropped_number_falls_back():
    out = {"noticed": ["open flowers", "buds"], "conditions": []}
    r = g.extract_observation("m", "petunia", "5 open flowers and 3 buds", ask=fake(out))
    assert r["parsed_by"] == "fallback"
    assert r["noticed"] == ["5 open flowers and 3 buds"]
    assert g.events_from_extract(r) == ["dropped_number"]


def test_unsupported_detail_is_removed_not_kept():
    out = {"noticed": ["3 yellow leaves", "aphid infestation"], "conditions": []}
    r = g.extract_observation("m", "basil", "3 yellow leaves", ask=fake(out))
    assert r["parsed_by"] == "gemma"
    assert r["noticed"] == ["3 yellow leaves"]
    assert "aphid infestation" in r["removed_unsupported"]
    assert g.events_from_extract(r) == ["unsupported_detail_removed"]


def test_reading_that_matches_nothing_falls_back():
    out = {"noticed": ["purple roses"], "conditions": []}
    r = g.extract_observation("m", "basil", "3 yellow leaves", ask=fake(out))
    assert r["parsed_by"] == "fallback"
    assert r["fallback_code"] == "reading_mismatch"


def test_wrong_types_do_not_crash():
    r = g.extract_observation("m", "basil", "3 yellow leaves", ask=fake({"noticed": "not a list"}))
    assert r["parsed_by"] == "fallback"


# ---------- look_first ----------
def test_look_first_survives_none_output():
    r = g.look_first("m", "basil", plant_with("observation"), ask=fake(None))
    assert r["decision"] == "LOOK_FIRST"
    assert r["source"] == "fallback"
    assert g.event_from_look(r) == "builtin_question_used"


def test_look_first_blocks_pest_question_not_in_notes():
    out = {"decision": "LOOK_FIRST", "question": "Are there aphids on the leaves?", "reason": "check"}
    r = g.look_first("m", "basil", plant_with("observation"), ask=fake(out))
    assert r["source"] == "guard"
    assert r["question"] == g.FALLBACK_QUESTION
    assert g.event_from_look(r) == "offtopic_question_blocked"


def test_look_first_allows_pest_question_when_gardener_mentioned_pests():
    out = {"decision": "LOOK_FIRST", "question": "Are there aphids under the leaves?", "reason": "check"}
    r = g.look_first("m", "basil", plant_with("observation", text="I saw aphids on the stem"), ask=fake(out))
    assert r["source"] == "gemma"


def test_gate_needs_two_notes_and_a_followup():
    out = {"decision": "ENOUGH_EVIDENCE", "question": "Do the shoots have buds?", "reason": "ok"}
    assert g.look_first("m", "x", plant_with("observation"), ask=fake(out))["decision"] == "LOOK_FIRST"
    assert g.look_first("m", "x", plant_with("observation", "observation"), ask=fake(out))["decision"] == "LOOK_FIRST"
    enough = plant_with("observation", "follow-up", "observation")
    assert g.look_first("m", "x", enough, ask=fake(out))["decision"] == "ENOUGH_EVIDENCE"


# ---------- propose_trial ----------
def test_propose_trial_rejects_bad_output():
    for bad in (None, {"choice": "fertilizer"}, {"choice": ""}):
        r = g.propose_trial("m", "x", plant_with("observation"), ask=fake(bad))
        assert r["trial_recommended"] is False
        assert g.event_from_trial(r) == "trial_pick_rejected"


def test_keep_watching_is_not_a_rejection():
    r = g.propose_trial("m", "x", plant_with("observation"), ask=fake({"choice": "keep_watching"}))
    assert r["trial_recommended"] is False
    assert g.event_from_trial(r) is None


def test_valid_pick_builds_trial_from_the_menu():
    r = g.propose_trial("m", "x", plant_with("observation"),
                        ask=fake({"choice": "Water_Less", "measure": "buds" * 50}))
    assert r["trial_recommended"] is True
    assert r["change"] == g.TRIAL_OPTIONS["water_less"]["change"]
    assert len(r["measure"]) <= 80
    assert r["chosen_by"] == "Gemma"


# ---------- verdicts ----------
def test_verdict_needs_two_days():
    assert g.verdict([rec("2026-10-01", 1, 1)]).startswith("Not enough data")


def test_verdict_without_comparison_is_weak():
    v = g.verdict([rec("2026-10-01", 1), rec("2026-10-04", 5)], has_control=False)
    assert "no comparison plant" in v


def test_verdict_with_comparison():
    v = g.verdict([rec("2026-10-01", 2, 2), rec("2026-10-04", 5, 3)])
    assert v.startswith("Promising, but not conclusive")
    v = g.verdict([rec("2026-10-01", 2, 2), rec("2026-10-04", 3, 3)])
    assert v.startswith("Inconclusive")
    v = g.verdict([rec("2026-10-01", 2, 2), rec("2026-10-04", 2, 6)])
    assert v.startswith("Opposite")


def test_observe_only_verdict_says_nothing_about_cause():
    v = g.verdict([rec("2026-10-05", 3), rec("2026-10-09", 7)], has_control=False, observe_only=True)
    assert v.startswith("Observation only")
    assert "says nothing about a cause" in v


# ---------- evidence ladder ----------
def test_level_0_for_observe_only_and_for_one_day():
    assert g.evidence_level(trial([rec("2026-10-05", 3), rec("2026-10-09", 7)],
                                  has_control=False, observe_only=True))["level"] == 0
    assert g.evidence_level(trial([rec("2026-10-05", 3, 3)]))["level"] == 0


def test_level_1_before_after_and_level_2_with_comparison():
    rs = [rec("2026-10-01", 2, 2), rec("2026-10-04", 5, 3)]
    assert g.evidence_level(trial(rs, has_control=False))["level"] == 1
    assert g.evidence_level(trial(rs))["level"] == 2


def test_weather_is_disclosed_but_does_not_lower_a_comparison_trial():
    rs = [rec("2026-10-01", 2, 2), rec("2026-10-04", 5, 3, changes=["weather"])]
    ev = g.evidence_level(trial(rs))
    assert ev["level"] == 2
    assert any("Rain or unusual weather" in n for n in ev["notes"])
    assert "reported" in g.trial_verdict(trial(rs))


def test_weather_without_comparison_cannot_be_separated():
    rs = [rec("2026-10-01", 2), rec("2026-10-04", 5, changes=["weather"])]
    ev = g.evidence_level(trial(rs, has_control=False))
    assert ev["level"] == 0
    assert ev["name"] == "Cannot separate the change"
    assert "inconclusive" in g.trial_verdict(trial(rs, has_control=False))


def test_another_change_caps_the_level_even_with_a_comparison():
    rs = [rec("2026-10-01", 2, 2, changes=["watering"]), rec("2026-10-04", 5, 3)]
    assert g.evidence_level(trial(rs))["level"] == 0


def test_start_changes_count_too():
    rs = [rec("2026-10-01", 2, 2), rec("2026-10-04", 5, 3)]
    assert g.evidence_level(trial(rs, start_changes=["moved"]))["level"] == 0


def _done(records, **kw):
    return trial(records, status="done", **kw)


def test_level_3_needs_two_finished_consistent_comparison_trials():
    a = _done([rec("2026-10-01", 2, 2), rec("2026-10-04", 6, 3)])
    b = _done([rec("2026-10-10", 1, 1), rec("2026-10-13", 5, 2)])
    assert g.evidence_level(a, [a, b])["level"] == 3
    assert g.evidence_level(b, [a, b])["level"] == 3


def test_level_3_not_reached_when_results_disagree_or_only_one_trial():
    a = _done([rec("2026-10-01", 2, 2), rec("2026-10-04", 6, 3)])
    b = _done([rec("2026-10-10", 1, 1), rec("2026-10-13", 1, 5)])
    assert g.evidence_level(a, [a, b])["level"] == 2
    assert g.evidence_level(a, [a])["level"] == 2


def test_level_3_not_reached_when_a_trial_had_other_changes_or_different_change():
    a = _done([rec("2026-10-01", 2, 2), rec("2026-10-04", 6, 3)])
    b = _done([rec("2026-10-10", 1, 1, changes=["moved"]), rec("2026-10-13", 5, 2)])
    c = _done([rec("2026-10-10", 1, 1), rec("2026-10-13", 5, 2)], change="Water one day less often")
    assert g.evidence_level(a, [a, b])["level"] == 2
    assert g.evidence_level(a, [a, c])["level"] == 2


def test_backfilled_entries_are_disclosed():
    rs = [rec("2026-10-05", 3, logged_on="2026-10-09"), rec("2026-10-09", 7, logged_on="2026-10-09")]
    ev = g.evidence_level(trial(rs, has_control=False, observe_only=True))
    assert any("backfilled" in n for n in ev["notes"])


# ---------- table and summary ----------
def test_table_shows_extra_columns_only_when_used():
    plain = g.table_rows([rec("2026-10-01", 1, 1), rec("2026-10-02", 2, 1)], True, "flowers")
    assert "also changed" not in plain[0] and "note" not in plain[0]
    rich = g.table_rows([rec("2026-10-01", 1, 1, changes=["weather"], note="side angle"),
                         rec("2026-10-02", 2, 1)], True, "flowers")
    assert rich[0]["also changed"] == "Rain or unusual weather"
    assert rich[0]["note"] == "side angle"
    assert rich[1]["also changed"] == "-"


def test_summary_markdown_includes_evidence_level():
    d = g.sample_garden()
    name = "Sample basil (illustrative)"
    md = g.summary_markdown(name, d["plants"][name])
    assert "Evidence level 2 of 3" in md
    assert "cannot prove a cause" in md


# ---------- sample garden ----------
def test_sample_garden_is_clearly_labelled_and_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "BASE_DIR", tmp_path)
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "garden.json")
    d = g.sample_garden()
    assert not (tmp_path / "garden.json").exists()
    assert all("illustrative" in n.lower() and p["sample"] for n, p in d["plants"].items())
    assert all(o["text"].startswith("[Sample]") for p in d["plants"].values() for o in p["observations"])
    json.dumps(d)  # must be serialisable


def test_sample_garden_shows_the_whole_loop():
    d = g.sample_garden()
    basil = d["plants"]["Sample basil (illustrative)"]
    t = basil["trials"][0]
    assert g.trial_verdict(t, basil["trials"]).startswith("Promising, but not conclusive")
    assert g.evidence_level(t, basil["trials"])["level"] == 2
    mint = d["plants"]["Sample mint (illustrative)"]
    assert mint["pending"]["question"] and mint["ready_for_trial"] is False


# ---------- guard events ----------
def test_guard_summary_counts_only_real_events():
    data = {"plants": {}}
    g.log_event(data, "dropped_number", "petunia")
    g.log_event(data, "dropped_number", "petunia")
    g.log_event(data, "made_up_event", "petunia")
    s = g.guard_summary(data)
    assert s == {g.EVENT_LABELS["dropped_number"]: 2}
    assert g.guard_summary({"plants": {}}) == {}


# ---------- storage and photos ----------
def test_save_and_load_roundtrip_and_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "BASE_DIR", tmp_path)
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "garden.json")
    data = {"plants": {}}
    g.get_plant(data, "marigold")["observations"].append({"date": "2026-10-09", "kind": "observation", "text": "7 open"})
    g.log_event(data, "unusable_output", "marigold")
    g.save_data(data)
    assert g.load_data() == data
    (tmp_path / "garden.json").write_text("{not json", encoding="utf-8")
    assert g.load_data() == {"plants": {}}


def test_save_photo_validates_and_shrinks(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "PHOTO_DIR", tmp_path / "photos")
    buf = io.BytesIO()
    Image.new("RGB", (3000, 2000), "orange").save(buf, "PNG")
    rel = g.save_photo("Marigold", "2026-10-09", "m.png", buf.getvalue())
    assert rel.startswith("photos/marigold-2026-10-09")
    assert max(Image.open(tmp_path / rel).size) <= g.PHOTO_MAX_SIDE
    with pytest.raises(ValueError):
        g.save_photo("m", "2026-10-09", "m.heic", b"x")
    with pytest.raises(ValueError):
        g.save_photo("m", "2026-10-09", "m.png", b"not an image")
    with pytest.raises(ValueError):
        g.save_photo("m", "2026-10-09", "m.png", b"")


# ---------- plant-aware menu ----------
def test_dry_loving_plants_do_not_get_water_more():
    assert "water_more" not in g.allowed_options("Bougainvillea cutting 1")
    assert "water_more" not in g.allowed_options("my bougainvillia")
    assert "water_more" in g.allowed_options("Marigold")
    assert "water_less" in g.allowed_options("Bougainvillea")  # only removes, never adds


def test_gemma_cannot_pick_an_unsuitable_option_for_the_plant():
    r = g.propose_trial("m", "Bougainvillea", plant_with("observation"), ask=fake({"choice": "water_more"}))
    assert r["trial_recommended"] is False
    assert g.event_from_trial(r) == "trial_pick_rejected"
    ok = g.propose_trial("m", "Marigold", plant_with("observation"), ask=fake({"choice": "water_more"}))
    assert ok["trial_recommended"] is True


def test_prompt_only_lists_allowed_options():
    seen = {}

    def spy(model, system, user):
        seen["system"] = system
        return {"choice": "keep_watching"}

    g.propose_trial("m", "Bougainvillea", plant_with("observation"), ask=spy)
    assert "water_more" not in seen["system"]
    assert "brighter" in seen["system"]


# ---------- repeat a trial ----------
def test_repeat_trial_is_a_fresh_trial_that_can_reach_level_3():
    a = _done([rec("2026-10-01", 2, 2), rec("2026-10-04", 6, 3)])
    a.update({"question": "q", "hypothesis": "h", "keep_same": "k", "comparison": "c", "start": "2026-10-01",
              "duration_days": 3})
    b = g.repeat_trial(a, start="2026-10-10")
    assert b["records"] == [] and b["status"] == "active" and b["change"] == a["change"]
    assert b["repeat_of"] == "2026-10-01" and b["start_changes"] == []
    a["records"].append(rec("2026-10-05", 9, 3))  # editing the old one must not touch the new one
    assert b["records"] == []
    b["records"] = [rec("2026-10-10", 1, 1), rec("2026-10-13", 5, 2)]
    b["status"] = "done"
    assert g.evidence_level(b, [a, b])["level"] == 3


# ---------- chart data ----------
def test_chart_series_needs_two_days_and_labels_plants():
    t = trial([rec("2026-10-01", 2, 2)])
    assert g.chart_series(t) == {}
    t = trial([rec("2026-10-04", 5, 3), rec("2026-10-01", 2, 2)])
    cs = g.chart_series(t)
    assert cs["index"] == ["2026-10-01", "2026-10-04"]
    assert cs["series"]["flowers (changed plant)"] == [2, 5]
    assert cs["series"]["flowers (comparison plant)"] == [2, 3]


def test_chart_series_observe_only_has_a_single_line():
    t = trial([rec("2026-10-05", 3), rec("2026-10-09", 7)], has_control=False, observe_only=True)
    assert list(g.chart_series(t)["series"]) == ["flowers"]


# ---------- plant details ----------
def test_summary_includes_details_the_gardener_gave():
    d = g.sample_garden()
    name = "Sample basil (illustrative)"
    d["plants"][name]["facts"] = "Mother plant also blooms. Same black bag."
    md = g.summary_markdown(name, d["plants"][name])
    assert "Details the gardener gave: Mother plant also blooms." in md


# ---------- safe loading ----------
def test_unreadable_file_is_moved_aside_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "BASE_DIR", tmp_path)
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "garden.json")
    (tmp_path / "garden.json").write_text("{broken", encoding="utf-8")
    data, notice = g.load_data_checked()
    assert data == {"plants": {}}
    assert "could not be read" in notice
    kept = list(tmp_path.glob("garden.unreadable-*.json"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == "{broken"
    assert g.load_data_checked() == ({"plants": {}}, "")  # a second load does not make more copies


def test_wrong_shape_file_is_also_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "BASE_DIR", tmp_path)
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "garden.json")
    (tmp_path / "garden.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert g.load_data_checked()[1] != ""
    assert len(list(tmp_path.glob("garden.unreadable-*.json"))) == 1


def test_missing_file_is_a_normal_empty_start(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "nothing.json")
    assert g.load_data_checked() == ({"plants": {}}, "")


def test_incomplete_plants_are_filled_in_so_the_app_cannot_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "BASE_DIR", tmp_path)
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "garden.json")
    raw = {"plants": {"rose": {"observations": [{"text": "bud"}], "trials": [{"records": [{"date": "2026-10-07"}]}]},
                      "junk": 5}}
    (tmp_path / "garden.json").write_text(json.dumps(raw), encoding="utf-8")
    d = g.load_data()
    assert list(d["plants"]) == ["rose"]
    p = d["plants"]["rose"]
    assert p["pending"] is None and p["ready_for_trial"] is False
    assert p["observations"][0]["kind"] == "observation" and p["observations"][0]["date"] == ""
    t = p["trials"][0]
    assert t["status"] == "cancelled" and t["records"][0]["changed"] == 0
    g.history_text(p)
    g.table_rows(t["records"], True, "flowers")
    g.summary_markdown("rose", p)


# ---------- backdated notes ----------
def test_notes_are_ordered_by_the_date_they_describe():
    plant = {"observations": [
        {"date": "2026-10-10", "kind": "observation", "text": "four open"},
        {"date": "2026-10-07", "kind": "observation", "text": "two open"}]}
    lines = g.history_text(plant).splitlines()
    assert "2026-10-07" in lines[0] and "2026-10-10" in lines[1]
    assert g.build_trial("brighter", "flowers", plant, chosen_by="Gemma")["based_on"] == "four open"


def test_note_is_late_only_when_logged_on_a_later_day():
    assert g.note_is_late({"date": "2026-10-07", "logged_on": "2026-10-10"})
    assert not g.note_is_late({"date": "2026-10-10", "logged_on": "2026-10-10"})
    assert not g.note_is_late({"date": "2026-10-10"})


# ---------- what is due today ----------
TODAY = __import__("datetime").date(2026, 10, 10)


def _garden(*, pending=None, trials=()):
    return {"plants": {"Petunia": {"observations": [], "pending": pending, "ready_for_trial": False,
                                   "trials": list(trials)}}}


def _active(records, **kw):
    t = trial(records, status="active", **kw)
    t.update({"duration_days": 3, "start": "2026-10-07"})
    return t


def test_nothing_due_is_an_empty_list():
    assert g.today_items(_garden(), on=TODAY) == []
    assert g.today_items(_garden(trials=[trial([rec("2026-10-07", 1)], status="done")]), on=TODAY) == []


def test_pending_question_is_listed():
    items = g.today_items(_garden(pending={"question": "Are the buds longer?"}), on=TODAY)
    assert items == [{"plant": "Petunia", "kind": "look", "text": "Go and look: Are the buds longer?"}]


def test_count_due_shows_day_last_count_and_comparison():
    t = _active([rec("2026-10-08", 2, 1)])
    text = g.today_items(_garden(trials=[t]), on=TODAY)[0]["text"]
    assert text.startswith("Day 2 of 3: count flowers")
    assert "comparison plant" in text and "Last count: 2 on 2026-10-08" in text


def test_counted_today_and_first_count_messages():
    done = g.today_items(_garden(trials=[_active([rec("2026-10-10", 4)])]), on=TODAY)[0]["text"]
    assert done == "Counted today. Nothing more to do."
    first = g.today_items(_garden(trials=[_active([])]), on=TODAY)[0]["text"]
    assert "BEFORE you change anything" in first
    watch = g.today_items(_garden(trials=[_active([], observe_only=True, has_control=False)]), on=TODAY)[0]["text"]
    assert watch == "Enter your first count."


def test_enough_days_logged_is_mentioned_and_bad_dates_do_not_crash():
    t = _active([rec("2026-10-05", 3), rec("2026-10-08", 5)], has_control=False)
    assert "enough days logged" in g.today_items(_garden(trials=[t]), on=TODAY)[0]["text"]
    bad = _active([rec("not-a-date", 3)], has_control=False)
    assert "count flowers" in g.today_items(_garden(trials=[bad]), on=TODAY)[0]["text"]


# ---------- next step ----------
def test_next_step_follows_the_evidence_level():
    assert "two different days" in g.next_step(trial([rec("2026-10-01", 1, 1)]))
    watch = trial([rec("2026-10-05", 3), rec("2026-10-09", 7)], has_control=False, observe_only=True)
    assert "pick one change" in g.next_step(watch)
    rs = [rec("2026-10-01", 2, 2), rec("2026-10-04", 5, 3)]
    assert "similar plant left unchanged" in g.next_step(trial(rs, has_control=False))
    assert "once more" in g.next_step(trial(rs))
    assert "nothing else changing" in g.next_step(trial(rs, start_changes=["moved"]))
    a = _done([rec("2026-10-01", 2, 2), rec("2026-10-04", 6, 3)])
    b = _done([rec("2026-10-10", 1, 1), rec("2026-10-13", 5, 2)])
    assert "not proof" in g.next_step(a, [a, b])


def test_summary_includes_next_step_and_collapses_note_newlines():
    d = g.sample_garden()
    name = "Sample basil (illustrative)"
    d["plants"][name]["observations"][0]["text"] = "line one\nline two"
    md = g.summary_markdown(name, d["plants"][name])
    assert "Next step:" in md and "line one line two" in md


# ---------- count check and photo helper ----------
def test_big_jumps_flags_only_large_moves():
    labels = {"changed": "flowers (changed plant)", "control": "flowers (comparison plant)"}
    assert g.big_jumps({"changed": 2, "control": 2}, {"changed": 12, "control": 3}, labels) == [
        "flowers (changed plant): 2 to 12"]
    assert g.big_jumps({"changed": 7, "control": 2}, {"changed": 5, "control": 2}, labels) == []
    assert g.big_jumps({"changed": 20}, {"changed": 14}, {"changed": "x"}) == []
    assert g.big_jumps({"changed": 0}, {"changed": 5}, {"changed": "x"}) == ["x: 0 to 5"]


def test_series_labels_follow_the_trial_shape():
    assert g.series_labels(trial([], has_control=False)) == {"changed": "flowers"}
    both = g.series_labels({"measure": "flowers", "measure2": "buds", "has_control": True})
    assert list(both) == ["changed", "control", "changed2", "control2"]


def test_latest_photo_picks_the_newest_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "PHOTO_DIR", tmp_path)
    (tmp_path / "a.jpg").write_bytes(b"x")
    t = trial([rec("2026-10-07", 1, photo="photos/a.jpg"), rec("2026-10-08", 2, photo="photos/gone.jpg")])
    date, path = g.latest_photo(t)
    assert date == "2026-10-07" and path.name == "a.jpg"
    assert g.latest_photo(trial([rec("2026-10-07", 1)])) is None


# ---------- Gemma switched off ----------
def test_plain_observation_keeps_exact_words_and_records_no_cause_or_events():
    r = g.plain_observation("Petunia", "2 open flowers, 4 buds")
    assert r["noticed"] == ["2 open flowers, 4 buds"] and r["cause"] == "unknown"
    assert r["parsed_by"] == "no_model" and g.events_from_extract(r) == []


def test_plain_look_first_always_asks_to_look_and_logs_nothing():
    r = g.plain_look_first()
    assert r["decision"] == "LOOK_FIRST" and r["question"] == g.FALLBACK_QUESTION
    assert g.event_from_look(r) is None


# ---------- pest guard matches whole words ----------
def test_ordinary_words_that_contain_a_pest_word_are_not_blocked():
    out = {"decision": "LOOK_FIRST", "question": "Is the light limited on the shady side?", "reason": "check"}
    r = g.look_first("m", "rose", plant_with("observation"), ask=fake(out))
    assert r["source"] == "gemma" and r["question"] == out["question"]


def test_plural_pest_words_are_still_blocked():
    for q in ("Do you see mites on the stems?", "Are there bugs under the leaves?", "Is there any fungus?"):
        out = {"decision": "LOOK_FIRST", "question": q, "reason": "check"}
        assert g.look_first("m", "rose", plant_with("observation"), ask=fake(out))["source"] == "guard"


# ---------- hand-edited counts and text ----------
def test_unreadable_counts_become_safe_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "BASE_DIR", tmp_path)
    monkeypatch.setattr(g, "DATA_FILE", tmp_path / "garden.json")
    records = [{"date": "2026-10-01", "changed": "x", "control": -4, "changed2": 99999, "control2": None},
               {"date": 20261002, "changed": 3.7}]
    raw = {"plants": {"rose": {"observations": [{"date": 5, "text": None}],
                               "trials": [{"records": records}]}}}
    (tmp_path / "garden.json").write_text(json.dumps(raw), encoding="utf-8")
    p = g.load_data()["plants"]["rose"]
    first, second = p["trials"][0]["records"]
    assert (first["changed"], first["control"], first["changed2"], first["control2"]) == (0, 0, g.COUNT_MAX, 0)
    assert second["changed"] == 3 and second["date"] == "20261002" and "changed2" not in second
    assert p["observations"][0]["date"] == "5" and p["observations"][0]["text"] == ""
    assert g.last_counts(p["trials"][0]["records"])["changed"] == 3
    g.table_rows(p["trials"][0]["records"], True, "flowers")


def test_last_counts_never_raises_on_bad_values():
    assert g.last_counts([{"date": "2026-10-01", "changed": "x", "control": float("inf")}])["changed"] == 0


# ---------- Ollama address ----------
@pytest.mark.parametrize("host, expected", [
    ("", "http://localhost:11434"),
    ("127.0.0.1:11500", "http://127.0.0.1:11500"),
    ("192.168.1.20", "http://192.168.1.20:11434"),
    ("http://myserver:8080/", "http://myserver:8080"),
    ("0.0.0.0:11434", "http://localhost:11434"),
    ("https://ollama.example.com", "https://ollama.example.com"),
    ("http://host:notaport", "http://localhost:11434"),
])
def test_ollama_address_follows_ollama_host(host, expected):
    assert g.ollama_base(host) == expected