import datetime as dt
import json

import pandas as pd
import streamlit as st

import logic as g

st.set_page_config(page_title="Garden Trials", page_icon="🌱")
st.title("🌱 Garden Trials")
st.caption("Look first. Change one thing. Let the garden answer.")

NEW_PLANT = "+ New plant"
ss = st.session_state

# ---------------- sidebar: model + sample garden ----------------
with st.sidebar:
    model = (st.text_input("Ollama model", g.DEFAULT_MODEL) or "").strip() or g.DEFAULT_MODEL
    if st.button("Check Ollama"):
        ok, msg = g.check_ollama(model)
        (st.success if ok else st.error)(msg)
    no_gemma = st.checkbox(
        "Run without Gemma", key="no_gemma",
        help="For trying the app with nothing installed but Python. Your notes are kept exactly as you write them, "
             "the app asks its built-in question, and you choose the change yourself.")
    if no_gemma:
        st.caption("Gemma is off. Nothing is read or suggested by a model.")
    else:
        st.caption("Notes are processed by a local open-weight Gemma model through Ollama. No cloud AI API is used.")
    st.caption("On your phone? With the phone on the same Wi-Fi, open the Network URL that Streamlit "
               "printed when it started. Keep this computer on.")
    st.divider()
    sample_mode = st.checkbox(
        "Try the illustrative sample garden",
        key="sample_mode",
        help="Loads made-up plants so you can see the whole loop. Nothing is saved to your real garden.",
    )
    if sample_mode and st.button("Reset sample garden"):
        ss["sample_data"] = g.sample_garden()

if sample_mode:
    if "sample_data" not in ss:
        ss["sample_data"] = g.sample_garden()
    data = ss["sample_data"]
    st.warning("SAMPLE GARDEN. Every plant, note, date and count here is made up to show how the app works. "
               "Nothing is saved to your real garden.")
else:
    data, load_notice = g.load_data_checked()
    if load_notice:
        ss["load_notice"] = load_notice
    if ss.get("load_notice"):
        st.warning(ss["load_notice"])


def save(d):
    """Real garden goes to garden.json; the sample garden stays in memory only."""
    if sample_mode:
        ss["sample_data"] = d
        return
    try:
        g.save_data(d)
    except OSError as e:
        st.error(f"Your change could not be saved ({e}). Check that the app's folder is writable.")
        st.stop()


with st.sidebar:
    with st.expander("How Garden Trials handles AI output"):
        st.markdown(
            "**Gemma proposes:** a reading of your note, a question to go and check, and one pick from a fixed menu of changes.\n\n"
            "**Python decides:** it removes details you did not write, keeps your exact words if the reading is unusable, "
            "never records a cause, blocks pest or disease questions you did not raise, builds every trial from the menu, "
            "and calculates the verdict and the evidence level.\n\n"
            "**The gardener decides:** whether to go and look, and what the counts really are."
        )
        summary = g.guard_summary(data)
        if summary:
            st.caption("Detected so far in this garden (only events the app actually noticed):")
            for label, count in summary.items():
                st.caption(f"{count} x {label}")
        else:
            st.caption("Nothing detected yet in this garden.")


def trial_table(t):
    return g.table_rows(t["records"], t.get("has_control", True), t.get("measure") or "count", t.get("measure2", ""))


def show_photos(t):
    m1, m2 = t.get("measure") or "count", t.get("measure2", "")
    shots = [(r, g.photo_path(r.get("photo"))) for r in sorted(t["records"], key=lambda r: r["date"])]
    shots = [(r, p) for r, p in shots if p]
    if shots:
        cols = st.columns(min(len(shots), 3))
        for idx, (r, p) in enumerate(shots):
            caption = f"{r['date']} · {m1}: {r.get('changed', 0)}" + (f" · {m2}: {r.get('changed2', 0)}" if m2 else "")
            try:
                cols[idx % len(cols)].image(str(p), caption=caption)
            except Exception:
                cols[idx % len(cols)].caption(f"{r['date']}: photo could not be shown")


def show_chart(t):
    cs = g.chart_series(t)
    if cs:
        st.line_chart(pd.DataFrame(cs["series"], index=cs["index"]))


def show_evidence(t, trials):
    ev = g.evidence_level(t, trials)
    st.markdown(f"**Evidence level {ev['level']} of 3: {ev['name']}**")
    for m in ev["missing"]:
        st.caption("Missing: " + m)
    for n in ev["notes"]:
        st.caption(n)
    st.markdown(f"**Next step:** {g.next_step(t, trials)}")
    st.caption("The level shows how strong the evidence is. It never means a cause has been proven.")


# ---------------- today ----------------
with st.expander("Today", expanded=True):
    due = g.today_items(data)
    if not due:
        st.caption("Nothing is due. Go and enjoy the garden.")
    for item in due:
        st.markdown(f"**{item['plant']}**: {item['text']}")

tab_obs, tab_trial, tab_hist = st.tabs(["1. Observe", "2. Trials", "3. History"])

# ---------------- Observe ----------------
with tab_obs:
    existing = sorted(data["plants"])
    options = existing + [NEW_PLANT]
    # After saving a note we jump to that plant and clear the note box. This has to happen
    # before the widgets below are created, because Streamlit locks a widget's value once it exists.
    goto = ss.pop("_goto_plant", None)
    if goto in options:
        ss["obs_plant"] = goto
        ss["obs_note"] = ""
        ss["obs_date"] = dt.date.today()
    if ss.get("obs_plant") not in options:
        ss.pop("obs_plant", None)

    flash = ss.pop("_flash", None)
    if flash:
        st.success("Saved.")
        st.markdown(flash["read_as"])
        for line in flash["notes"]:
            st.caption(line)

    choice = st.selectbox("Plant", options, key="obs_plant")
    if choice == NEW_PLANT:
        typed = " ".join(st.text_input("New plant name", key="new_plant").split())[:60]
        # reuse an existing plant if the name only differs by capital letters
        name = next((k for k in data["plants"] if k.lower() == typed.lower()), typed)
    else:
        name = choice
    note = st.text_area("What did you notice? (type it, or speak it with your phone keyboard mic)",
                        key="obs_note", height=100)
    note_day = st.date_input("When did you see this?", max_value=dt.date.today(), key="obs_date")

    if st.button("Record observation", type="primary"):
        if not name or name.lower() == NEW_PLANT.lower() or not note.strip():
            st.error("Add a plant name and a note.")
        else:
            created = name not in data["plants"]
            try:
                with st.spinner("Reading your note... (with Gemma this can take up to a minute on a laptop CPU)"):
                    if no_gemma:
                        structured = g.plain_observation(name, note.strip())
                    else:
                        structured = g.extract_observation(model, name, note.strip())
                    plant = g.get_plant(data, name)
                    entry = {"date": note_day.isoformat(), "kind": "observation", "text": note.strip(),
                             "structured": structured, "logged_on": g.today()}
                    plant["observations"].append(entry)
                    try:
                        lf = g.plain_look_first() if no_gemma else g.look_first(model, name, plant)
                    except g.OllamaError:
                        plant["observations"].remove(entry)
                        raise
            except g.OllamaError as e:
                if created:
                    data["plants"].pop(name, None)
                st.error(str(e) + " Or tick 'Run without Gemma' in the sidebar to carry on without it.")
            else:
                for kind in g.events_from_extract(structured):
                    g.log_event(data, kind, name)
                look_event = g.event_from_look(lf)
                if look_event:
                    g.log_event(data, look_event, name)
                if lf["decision"] == "LOOK_FIRST":
                    plant["pending"] = lf
                    plant["ready_for_trial"] = False
                else:
                    plant["pending"] = None
                    plant["ready_for_trial"] = True
                save(data)
                notes = []
                if structured.get("uncaptured"):
                    notes.append("Gemma's reading left out: " + ", ".join(structured["uncaptured"])
                                 + ". Your full note is saved exactly as you wrote it.")
                if structured.get("removed_unsupported"):
                    notes.append("Gemma added details you did not write, so the app removed them: "
                                 + "; ".join(structured["removed_unsupported"]))
                if structured.get("parsed_by") == "fallback":
                    notes.append(structured.get("fallback_reason", "Gemma's reading was not usable")
                                 + ", so your note was kept exactly as you wrote it.")
                if note_day.isoformat() != g.today():
                    notes.append("This note is dated " + note_day.isoformat() + " and marked as logged later.")
                if structured.get("parsed_by") == "no_model":
                    read_as = "**Your note was kept exactly as you wrote it.** Gemma is off."
                else:
                    read_as = ("**Gemma read this as:** " + "; ".join(map(str, structured["noticed"]))
                               + (" | **conditions:** " + "; ".join(map(str, structured["conditions"]))
                                  if structured["conditions"] else " | **conditions:** none mentioned"))
                ss["_flash"] = {"read_as": read_as, "notes": notes}
                ss["_goto_plant"] = name
                st.rerun()

    if name and name in data["plants"]:
        plant = data["plants"][name]
        with st.expander("Details you know about this plant (optional)"):
            st.caption("For example: which mother plant it came from, its age, pot or bag. "
                       "Kept for you and your summary. Gemma never sees it, so it cannot turn it into a cause.")
            facts = st.text_area("Details", plant.get("facts", ""), key=f"facts|{name}", height=70)
            if st.button("Save details", key=f"savefacts|{name}"):
                plant["facts"] = facts.strip()[:500]
                save(data)
                st.success("Saved.")
        pending = plant.get("pending")
        if pending and pending.get("question"):
            st.warning("👀 LOOK FIRST. No advice until you've checked.")
            st.markdown(f"**{pending['question']}**")
            src = pending.get("source")
            st.caption("Look before changing anything." if src == "gemma" else pending.get("reason", ""))
            st.caption("Question written by: " + (
                "Gemma" if src == "gemma" else "the sample garden" if src == "sample"
                else "the app's built-in question (Gemma is switched off)" if src == "builtin"
                else "the app's built-in question (Gemma's question was about pests or disease, which your notes did not mention)"
                if src == "guard"
                else "the app's built-in question (Gemma's wasn't usable)"))
            found = st.text_area("Go look at the plant. What did you find?", key=f"found|{name}", height=80)
            found_day = st.date_input("When did you look?", value=dt.date.today(), max_value=dt.date.today(),
                                      key=f"fdate|{name}")
            if st.button("Save what I found"):
                if len(found.strip()) >= 15:
                    plant["observations"].append({"date": found_day.isoformat(), "kind": "follow-up",
                                                  "text": found.strip(), "question": pending["question"],
                                                  "question_source": pending.get("source", ""),
                                                  "logged_on": g.today()})
                    plant["pending"] = None
                    plant["ready_for_trial"] = True
                    save(data)
                    st.rerun()
                else:
                    st.error("Go look, then write a sentence about what you found (at least 15 characters).")
        elif plant["observations"]:
            st.success("Nothing pending. Head to the Trials tab when you're ready.")

# ---------------- Trials ----------------
with tab_trial:
    plants = sorted(data["plants"])
    if not plants:
        st.info("Record an observation first.")
    else:
        if ss.get("trial_plant") not in plants:
            ss.pop("trial_plant", None)
        sel = st.selectbox("Plant", plants, key="trial_plant")
        p = data["plants"][sel]
        has_active = any(t.get("status") == "active" for t in p["trials"])
        if not p.get("ready_for_trial"):
            st.info("Look First: answer the pending question in the Observe tab before a trial can be suggested.")
        else:
            if has_active:
                st.info("One trial at a time. Finish or cancel the active trial below to start another.")
            if st.button("Suggest a trial", disabled=has_active or no_gemma):
                try:
                    with st.spinner("Gemma is choosing a small experiment..."):
                        proposal = g.propose_trial(model, sel, p)
                    trial_event = g.event_from_trial(proposal)
                    if trial_event:
                        g.log_event(data, trial_event, sel)
                        save(data)
                    ss["proposal_" + sel] = proposal
                except g.OllamaError as e:
                    st.error(str(e))
            if no_gemma:
                st.caption("Gemma is off, so choose the one change yourself below.")
            c_a, c_b = st.columns(2)
            watch_measure = c_a.text_input("What will you count?", "open flowers",
                                           key=f"wm|{sel}", disabled=has_active)
            watch_measure2 = c_b.text_input("Also count (optional)", "buds", key=f"wm2|{sel}", disabled=has_active)
            options_menu = g.allowed_options(sel)
            pick = st.selectbox("Or choose the one change yourself", options_menu,
                                format_func=lambda k: g.TRIAL_LABELS[k], key=f"pick|{sel}", disabled=has_active)
            if len(options_menu) < len(g.TRIAL_OPTIONS):
                st.caption("The menu leaves out changes that run against how this kind of plant usually does best.")
            if st.button("Use this change", key=f"usepick|{sel}", disabled=has_active):
                ss["proposal_" + sel] = g.build_trial(pick, watch_measure, p, chosen_by="you")
            if st.button("Or just watch: log daily counts without changing anything", key=f"watch|{sel}",
                         disabled=has_active):
                m1 = watch_measure.strip() or "open flowers"
                m2 = watch_measure2.strip()
                if m2.lower() == m1.lower():
                    m2 = ""
                p["trials"].append({"question": "How does this plant change on its own over a few days?",
                                    "hypothesis": "No change made", "change": "Nothing (observe only)",
                                    "keep_same": "everything", "comparison": "none",
                                    "measure": m1, "measure2": m2, "duration_days": g.TRIAL_DAYS,
                                    "start": g.today(), "status": "active", "records": [],
                                    "has_control": False, "observe_only": True})
                ss.pop("proposal_" + sel, None)
                save(data)
                st.rerun()
            prop = ss.get("proposal_" + sel)
            if prop and not has_active:
                if not prop["trial_recommended"]:
                    st.info(prop["note"])
                else:
                    st.subheader("Proposed trial")
                    if prop.get("chosen_by") == "you":
                        st.caption("You chose this change.")
                    elif prop.get("based_on"):
                        st.caption("Gemma picked this from the menu, using your latest note: " + prop["based_on"])
                    st.markdown(
                        f"**Question:** {prop['question']}  \n**Hypothesis:** {prop['hypothesis']}  \n"
                        f"**Change (one thing):** {prop['change']}  \n**Keep the same:** {prop['keep_same']}  \n"
                        f"**Comparison:** {prop['comparison']}  \n**Measure:** {prop['measure']}  \n"
                        f"**Duration:** {prop['duration_days']} days"
                    )
                    prop_measure2 = st.text_input("Also count (optional)", "", key=f"pm2|{sel}")
                    has_ctrl = st.checkbox("I have a similar plant to leave unchanged as a comparison", value=True,
                                           key=f"hc|{sel}")
                    start_changes = st.multiselect(
                        "Besides this one change, is anything else different right now? (optional)",
                        list(g.CHANGE_FLAGS), format_func=lambda k: g.CHANGE_FLAGS[k], key=f"sc|{sel}")
                    st.caption("Enter your baseline count BEFORE you change anything.")
                    if st.button("Start this trial"):
                        m2 = prop_measure2.strip()
                        if m2.lower() == prop["measure"].strip().lower():
                            m2 = ""
                        p["trials"].append({**prop, "measure2": m2, "start": g.today(), "status": "active",
                                            "records": [], "has_control": has_ctrl,
                                            "start_changes": list(start_changes)})
                        ss.pop("proposal_" + sel, None)
                        save(data)
                        st.rerun()

        for i, t in enumerate(p["trials"]):
            if t.get("status") != "active":
                continue
            n = len(t["records"])  # part of the widget keys, so inputs reset after every add or undo
            key = lambda kind: f"{kind}|{sel}|{i}|{n}"
            m1, m2 = t.get("measure") or "count", t.get("measure2", "")
            hc = t.get("has_control", True)
            last = g.last_counts(t["records"])
            st.divider()
            st.subheader(f"Active trial: {t['question']}")
            st.caption(f"Started {t['start']}. Counting: {m1}" + (f" and {m2}" if m2 else "") + f". Change: {t['change']}")
            c1, c2, c3 = st.columns(3)
            day = c1.date_input("Date", key=key("date"), max_value=dt.date.today())
            changed, control, changed2, control2 = 0, 0, 0, 0
            if t.get("observe_only"):
                changed = c2.number_input(f"{m1}: count today", 0, 1000, last["changed"], key=key("n1"))
                if m2:
                    changed2 = c3.number_input(f"{m2}: count today", 0, 1000, last["changed2"], key=key("n2"))
            else:
                changed = c2.number_input(f"{m1}: plant you changed", 0, 1000, last["changed"], key=key("n1"))
                if hc:
                    control = c3.number_input(f"{m1}: comparison plant", 0, 1000, last["control"], key=key("k1"))
                if m2:
                    _, e2, e3 = st.columns(3)
                    changed2 = e2.number_input(f"{m2}: plant you changed", 0, 1000, last["changed2"], key=key("n2"))
                    if hc:
                        control2 = e3.number_input(f"{m2}: comparison plant", 0, 1000, last["control2"], key=key("k2"))
            st.caption("Count only what you can see clearly. If you are unsure, say so in the note below.")

            jumps = []
            if t["records"]:
                jumps = g.big_jumps(last, {"changed": changed, "control": control,
                                           "changed2": changed2, "control2": control2}, g.series_labels(t))
            count_ok = True
            if jumps:
                st.warning("Big change since the last entry: " + "; ".join(jumps) + ".")
                count_ok = st.checkbox("I counted again. These numbers are right.", key=key("jumpok"))

            if sample_mode:
                photo = None
                st.caption("Photo upload is switched off in the sample garden.")
            else:
                previous = g.latest_photo(t)
                if previous:
                    st.image(str(previous[1]), width=160,
                             caption=f"Last photo ({previous[0]}). Try to take today's from the same spot and angle.")
                photo = st.file_uploader("Photo today (optional). Same spot, angle and distance each time.",
                                         type=list(g.PHOTO_TYPES),
                                         key=key("photo"))
            rec_note = st.text_input("Note about this entry (optional), for example: photo taken from a different angle",
                                     key=key("note"))
            rec_changes = []
            if not t.get("observe_only"):
                rec_changes = st.multiselect(
                    "Besides your one change, did anything else change since the last entry? (optional)",
                    list(g.CHANGE_FLAGS), format_func=lambda k: g.CHANGE_FLAGS[k], key=key("chg"))
            st.caption("Your first entry is the baseline (day 0). Counts start at your last saved values, "
                       "so change them to what you actually see. Log each day once. "
                       "If you enter an earlier date, the app marks it as backfilled.")
            if st.button("Add measurement", key=f"add|{sel}|{i}", disabled=not count_ok):
                if any(r["date"] == day.isoformat() for r in t["records"]):
                    st.error("That date is already logged. Undo the last measurement if it was a mistake.")
                else:
                    rec = {"date": day.isoformat(), "changed": int(changed), "control": int(control),
                           "logged_on": g.today()}
                    if m2:
                        rec["changed2"], rec["control2"] = int(changed2), int(control2)
                    if rec_changes:
                        rec["changes"] = list(rec_changes)
                    if rec_note.strip():
                        rec["note"] = rec_note.strip()[:200]
                    err = None
                    if photo is not None:
                        try:
                            rec["photo"] = g.save_photo(sel, rec["date"], photo.name, photo.getvalue())
                        except (ValueError, OSError) as e:
                            err = str(e)
                    if err:
                        st.error(err)
                    else:
                        t["records"].append(rec)
                        save(data)
                        st.rerun()
            if not t["records"]:
                st.caption("No measurements yet.")
            else:
                st.table(trial_table(t))
                show_chart(t)
                show_photos(t)
                if st.button("Undo last measurement", key=f"undo|{sel}|{i}"):
                    t["records"].pop()
                    save(data)
                    st.rerun()
                st.info(g.trial_verdict(t, p["trials"]))
                show_evidence(t, p["trials"])
                surprise = st.text_area("What surprised you? (optional)", key=f"surprise|{sel}|{i}", height=60)
                can_finish = g.distinct_days(t["records"]) >= 2
                if not can_finish:
                    st.caption("Log at least two different days to finish this trial.")
                if st.button("Finish trial", key=f"fin|{sel}|{i}", disabled=not can_finish):
                    t["status"] = "done"
                    t["surprise"] = surprise.strip()
                    save(data)
                    st.rerun()
            if st.button("Cancel this trial", key=f"cancel|{sel}|{i}"):
                t["status"] = "cancelled"
                save(data)
                st.rerun()

        # Repeating a finished trial starts a NEW trial; results are never merged into the old one.
        if not has_active:
            for i in reversed(range(len(p["trials"]))):
                t = p["trials"][i]
                if t.get("status") == "done" and not t.get("observe_only"):
                    st.divider()
                    st.caption(f"Finished trial from {t.get('start', '')}: {t.get('change', '')}")
                    if st.button("Repeat this trial (new baseline, new comparison)", key=f"repeat|{sel}|{i}"):
                        p["trials"].append(g.repeat_trial(t))
                        save(data)
                        st.rerun()
                    break  # one repeat button (for the latest finished trial) keeps the tab uncluttered

# ---------------- History ----------------
with tab_hist:
    if not data["plants"]:
        st.info("Nothing yet.")
    for pname in sorted(data["plants"]):
        pl = data["plants"][pname]
        with st.expander(f"🌿 {pname}  ({len(pl['observations'])} notes, {len(pl['trials'])} trials)"):
            if pl.get("facts"):
                st.caption("Details you gave: " + pl["facts"])
            for o in sorted(pl["observations"], key=lambda o: o.get("date", "")):
                later = " (logged later)" if g.note_is_late(o) else ""
                st.markdown(f"- **{o['date']}** · {o['kind']}{later}: {o['text']}")
            for t in pl["trials"]:
                st.markdown(f"**Trial ({t.get('status', '')}):** {t.get('question', '')}")
                if t["records"]:
                    st.table(trial_table(t))
                    show_chart(t)
                    show_photos(t)
                    st.caption(g.trial_verdict(t, pl["trials"]))
                    show_evidence(t, pl["trials"])
                else:
                    st.caption("No measurements yet.")
                if t.get("surprise"):
                    st.caption("What surprised me: " + t["surprise"])
            st.download_button("Download this plant's summary (Markdown)", g.summary_markdown(pname, pl),
                               f"{g._slug(pname)}-summary.md", mime="text/markdown", key=f"md|{pname}")
    st.download_button("Download my garden data (JSON)", json.dumps(data, indent=2, ensure_ascii=False),
                       "garden.json", mime="application/json")