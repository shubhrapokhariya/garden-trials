"""Garden Trials core logic (no UI).

Gemma (served locally by Ollama) does three jobs:
  1. turn a gardener's note into a structured observation
  2. decide what to LOOK AT next (Look First gate)
  3. pick one tiny, one-variable experiment from a fixed menu

Everything else (saving, rules, verdicts, evidence levels) is plain Python so a
small model cannot break it.
"""
import io
import json
import os
import re
import datetime as dt
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageOps

# Data lives next to the app by default. Set GARDEN_DIR to keep a separate set
# (for example a demo folder) without touching your own notes and photos.
BASE_DIR = Path(os.environ.get("GARDEN_DIR") or Path(__file__).parent).resolve()
DATA_FILE = BASE_DIR / "garden.json"
PHOTO_DIR = BASE_DIR / "photos"
PHOTO_TYPES = ("jpg", "jpeg", "png", "webp")
MAX_PHOTO_BYTES = 25 * 1024 * 1024
PHOTO_MAX_SIDE = 1600



def ollama_base(host=None):
    """Address of the Ollama server. Honours OLLAMA_HOST, the same variable Ollama itself reads."""
    host = (os.environ.get("OLLAMA_HOST") if host is None else host) or ""
    host = host.strip().rstrip("/")
    if not host:
        return "http://localhost:11434"
    if "://" not in host:
        host = "http://" + host
    try:
        parts = urlsplit(host)
        port = parts.port
    except ValueError:  # not a usable address, so use Ollama's default
        return "http://localhost:11434"
    name = parts.hostname or "localhost"
    if name in ("0.0.0.0", "::"):  # a server bound to every address is reached on localhost
        name = "localhost"
    if ":" in name:  # bare IPv6 address
        name = f"[{name}]"
    port = port or (11434 if parts.scheme == "http" else None)
    return f"{parts.scheme}://{name}" + (f":{port}" if port else "")


OLLAMA_URL = ollama_base() + "/api/chat"
OLLAMA_TAGS = ollama_base() + "/api/tags"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL") or "gemma3:1b"

FALLBACK_QUESTION = (
    "Count the open flowers and the buds right now. Do the newest shoots have buds?"
)
OFF_TOPIC_WORDS = ("disease", "pest", "insect", "fungus", "fungal", "bug", "mite", "aphid")
# Whole-word match, so an ordinary word such as "limited" is not mistaken for "mite".
OFF_TOPIC_RE = re.compile(r"\b(?:" + "|".join(OFF_TOPIC_WORDS) + r")\w*", re.IGNORECASE)
STOPWORDS = {"and", "the", "with", "are", "was", "has", "have", "its", "but", "for", "not",
             "very", "some", "this", "that", "from", "into", "than", "then"}

EXTRACT_SYS = (
    "You turn a gardener's note into JSON. Use ONLY facts the gardener stated. "
    "Never guess a cause. Return JSON with keys: "
    '"plant" (string), "noticed" (list of short strings), '
    '"conditions" (list of short strings), "comparison" (string, empty if none). '
    'Put what the gardener SAW on the plant (flowers, shoots, leaves, buds, colours, counts) in "noticed". '
    'Put ONLY growing conditions (sunlight, watering, soil, location, weather) in "conditions"; '
    "use an empty list if none were stated. "
    'Example: note "3 yellow leaves, stem looks thin" gives '
    '{"plant": "basil", "noticed": ["3 yellow leaves", "stem looks thin"], '
    '"conditions": [], "comparison": ""}.'
)

LOOK_SYS = (
    "You help a gardener observe before acting. You get a plant's note history. "
    "Do NOT diagnose and do NOT recommend any change. Decide whether the gardener "
    'should look at the plant again. Return JSON: "decision" ("LOOK_FIRST" or '
    '"ENOUGH_EVIDENCE"), "question" (ONE specific thing to physically check on '
    'the plant, one sentence, linked to what the gardener wrote), "reason" (one short sentence). '
    'Good questions: "Do the new shoots have flower buds?" or "Is the soil still damp 2 cm below the surface?". '
    "Do not ask about disease or pests unless the gardener mentioned them."
)

TRIAL_OPTIONS = {
    "brighter": {
        "change": "Move the plant to a spot with more morning sunlight",
        "keep_same": "watering, pot, soil; no fertilizer or sprays",
        "hypothesis": "May produce more growth or flowers with more morning light",
    },
    "shadier": {
        "change": "Move the plant to a shadier spot (less afternoon sun)",
        "keep_same": "watering, pot, soil; no fertilizer or sprays",
        "hypothesis": "May do better with less direct afternoon sun",
    },
    "water_less": {
        "change": "Water one day less often than you do now",
        "keep_same": "position, pot, soil, amount of water each time; no fertilizer or sprays",
        "hypothesis": "May do better with slightly less frequent watering",
    },
    "water_more": {
        "change": "Water one day more often than you do now",
        "keep_same": "position, pot, soil, amount of water each time; no fertilizer or sprays",
        "hypothesis": "May do better with slightly more frequent watering",
    },
    "spacing": {
        "change": "Give the crowded plants more space (thin or separate them)",
        "keep_same": "position, watering, soil; no fertilizer or sprays",
        "hypothesis": "May grow better with less crowding",
    },
}
TRIAL_LABELS = {
    "brighter": "Move to more morning sun",
    "shadier": "Move to a shadier spot",
    "water_less": "Water one day less often",
    "water_more": "Water one day more often",
    "spacing": "Give crowded plants more space",
}
KEEP_WATCHING = "keep_watching"

# Plants that are usually kept on the dry side. For these the menu leaves out "water more", so neither
# Gemma nor the gardener is offered a change that runs against how the plant normally behaves.
# Plants not listed get the full menu. This only ever removes options, it never adds any.
DRY_LOVING = ("bougainvillea", "bougainvillia", "cactus", "cacti", "succulent", "aloe", "jade")
NOT_FOR_DRY = {"water_more"}
TRIAL_DAYS = 3
WAIT_NOTE = "Change nothing yet. Keep watching and look again after your next note."
COMPARISON_TEXT = (
    "A similar plant left exactly as it is. If you have none, compare the same plant "
    "before vs after and treat it as weaker evidence."
)

def allowed_options(plant_name):
    """Menu ids that make sense for this plant (a fixed allow-list, never decided by Gemma)."""
    low = str(plant_name).lower()
    if any(w in low for w in DRY_LOVING):
        return [k for k in TRIAL_OPTIONS if k not in NOT_FOR_DRY]
    return list(TRIAL_OPTIONS)


def trial_sys(options=None):
    return (
        "You help a gardener choose ONE small experiment, or choose to wait. You get a "
        "plant's note history. Pick exactly one option id from this list: "
        + ", ".join(list(options or TRIAL_OPTIONS) + [KEEP_WATCHING])
        + ". Use keep_watching if the history is thin or unclear. Never suggest "
        "fertilizer, sprays or chemicals. Return JSON: "
        '"choice" (one option id) and "measure" (something the gardener can COUNT, such as '
        "open flowers, buds or new leaves)."
    )


TRIAL_SYS = trial_sys()

# "What else changed?" flags, asked besides the ONE change being tested.
CHANGE_FLAGS = {
    "weather": "Rain or unusual weather",
    "moved": "Plant moved to a different spot (other than the tested change)",
    "watering": "Watering changed (other than the tested change)",
    "other": "Something else changed",
}
SHARED_FLAGS = {"weather"}  # hits the changed and the comparison plant alike

LEVEL_NAMES = {
    0: "Observation only",
    1: "Before and after",
    2: "With a comparison plant",
    3: "Repeated, results agree",
}

# Things the app can actually detect when it checks Gemma's output.
EVENT_LABELS = {
    "unusable_output": "Gemma's reading of a note was unusable (invalid or missing fields)",
    "reading_mismatch": "Gemma's reading did not match the note",
    "dropped_number": "Gemma's reading dropped a number from the note",
    "unsupported_detail_removed": "A detail Gemma added that the gardener did not write was removed",
    "builtin_question_used": "Gemma's Look First question was unusable, so the built-in question was used",
    "offtopic_question_blocked": "A pest or disease question not based on the notes was blocked",
    "trial_pick_rejected": "Gemma's trial choice was not on the menu, so no trial was proposed",
}


class OllamaError(Exception):
    pass


COUNT_KEYS = ("changed", "control", "changed2", "control2")
COUNT_MAX = 1000  # the largest count the entry form accepts


def _count(value):
    """A saved count as a whole number from 0 to COUNT_MAX. Anything unreadable becomes 0."""
    try:
        return max(0, min(int(value), COUNT_MAX))
    except (TypeError, ValueError, OverflowError):
        return 0


# ---------- storage ----------
def _normalise(raw):
    """Fill in anything missing so a hand-edited or older file cannot crash the app."""
    plants = {}
    for name, plant in raw["plants"].items():
        if not isinstance(plant, dict):
            continue
        plant["observations"] = [o for o in plant.get("observations") or [] if isinstance(o, dict)]
        plant["trials"] = [t for t in plant.get("trials") or [] if isinstance(t, dict)]
        plant.setdefault("pending", None)
        plant.setdefault("ready_for_trial", False)
        for o in plant["observations"]:
            for field, default in (("date", ""), ("kind", "observation"), ("text", "")):
                o[field] = str(o.get(field) or default)
        for t in plant["trials"]:
            t["records"] = [r for r in t.get("records") or [] if isinstance(r, dict)]
            for r in t["records"]:
                r["date"] = str(r.get("date") or "")
                for key in COUNT_KEYS:
                    if key in ("changed", "control") or key in r:
                        r[key] = _count(r.get(key, 0))
            t.setdefault("status", "cancelled")
            t.setdefault("question", "")
            t.setdefault("change", "")
            t.setdefault("start", "")
        plants[str(name)] = plant
    raw["plants"] = plants
    return raw


def load_data_checked():
    """Return (data, notice). A file that cannot be read is moved aside, never overwritten."""
    if not DATA_FILE.exists():
        return {"plants": {}}, ""
    try:
        loaded = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("plants"), dict):
            return _normalise(loaded), ""
    except (ValueError, OSError):  # includes bad JSON and bad encoding
        pass
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    aside = DATA_FILE.with_name(f"{DATA_FILE.stem}.unreadable-{stamp}{DATA_FILE.suffix}")
    try:
        DATA_FILE.replace(aside)
        notice = (f"{DATA_FILE.name} could not be read, so it was kept as {aside.name} "
                  "and a new, empty garden was started.")
    except OSError:
        notice = (f"{DATA_FILE.name} could not be read and could not be moved aside. "
                  "Please check the file before adding anything.")
    return {"plants": {}}, notice


def load_data():
    return load_data_checked()[0]


def save_data(data):
    """Write to a temp file first, then swap it in, so a crash can't leave half a file."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_name(DATA_FILE.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DATA_FILE)


def today():
    return dt.date.today().isoformat()


def get_plant(data, name):
    return data["plants"].setdefault(
        name, {"observations": [], "pending": None, "ready_for_trial": False, "trials": []}
    )


def history_text(plant, n=6):
    lines = []
    for o in sorted(plant["observations"], key=lambda o: o.get("date", ""))[-n:]:
        extra = ""
        conds = (o.get("structured") or {}).get("conditions") or []
        if conds:
            extra = " (conditions: " + "; ".join(map(str, conds)) + ")"
        lines.append(f"- {o['date']} [{o['kind']}] {o['text']}{extra}")
    return "\n".join(lines) or "(no notes yet)"


# ---------- Ollama ----------
def check_ollama(model):
    try:
        models = requests.get(OLLAMA_TAGS, timeout=5).json().get("models", [])
        names = [m.get("name") or m.get("model") or "" for m in models]
    except (requests.RequestException, ValueError, AttributeError):
        return False, "Ollama is not running. Start the Ollama app, then try again."
    if not any(n == model or n.startswith(model + ":") or n.split(":")[0] == model for n in names):
        return False, f"Ollama is running but '{model}' is not downloaded. Run: ollama pull {model}"
    return True, f"Ollama is running and '{model}' is ready."


def parse_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def ask_gemma(model, system, user):
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if r.status_code == 404:
            raise OllamaError(f"Model '{model}' is not downloaded. Run: ollama pull {model}")
        r.raise_for_status()
        return parse_json(r.json()["message"]["content"])
    except OllamaError:
        raise
    except requests.ConnectionError:
        raise OllamaError("Cannot reach Ollama. Start the Ollama app and try again.")
    except requests.Timeout:
        raise OllamaError("Ollama took too long to answer. Try again, or try a smaller model.")
    except requests.RequestException as e:
        raise OllamaError(f"Ollama error: {e}")
    except (KeyError, ValueError, TypeError):
        return None


# ---------- the three Gemma jobs ----------
def _numbers(text):
    return set(re.findall(r"\d+", text))


def _supported(item, text):
    """True if the item shares a meaningful word with what the gardener actually wrote."""
    words = [w for w in re.findall(r"[a-z]+", str(item).lower()) if len(w) > 3]
    if not words:
        return True
    low = text.lower()
    return any(w in low for w in words)


def _stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def uncaptured_words(text, out):
    """Words from the gardener's note that Gemma's reading did not carry over."""
    blob = " ".join(map(str, out.get("noticed", []))) + " " + " ".join(map(str, out.get("conditions", [])))
    blob = (blob + " " + str(out.get("comparison", ""))).lower()
    seen, missing = set(), []
    for w in re.findall(r"[a-z]+", text.lower()):
        if len(w) < 3 or w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        if _stem(w) not in blob:
            missing.append(w)
    return missing


def extract_observation(model, plant_name, text, ask=ask_gemma):
    out = ask(model, EXTRACT_SYS, f"Plant: {plant_name}\nNote: {text}")
    shape_ok = (
        isinstance(out, dict)
        and isinstance(out.get("noticed"), list)
        and isinstance(out.get("conditions", []), list)
        and len(out.get("noticed", [])) > 0
    )
    ok = shape_ok
    reason = "" if shape_ok else "Gemma's output was not usable"
    code = "" if shape_ok else "unusable_output"
    removed = []

    if ok:
        out.setdefault("conditions", [])
        # drop anything the gardener did not actually write
        for key in ("noticed", "conditions"):
            kept = [x for x in out[key] if _supported(x, text)]
            removed += [str(x) for x in out[key] if not _supported(x, text)]
            out[key] = kept
        if not out["noticed"]:
            ok = False
            reason = "Gemma's reading did not match your note"
            code = "reading_mismatch"

    if ok:
        # a condition should not also be listed as something noticed
        cond_low = {str(c).strip().lower() for c in out["conditions"]}
        deduped = [x for x in out["noticed"] if str(x).strip().lower() not in cond_low]
        if deduped:
            out["noticed"] = deduped
        got = " ".join(map(str, out["noticed"])) + " " + " ".join(map(str, out["conditions"]))
        if _numbers(text) - _numbers(got):
            ok = False
            reason = "Gemma's reading dropped a number from your note"
            code = "dropped_number"

    if not ok:
        out = {"noticed": [text], "conditions": [], "comparison": "", "parsed_by": "fallback",
               "fallback_reason": reason, "fallback_code": code, "uncaptured": [],
               "removed_unsupported": removed}
    else:
        out["parsed_by"] = "gemma"
        out["removed_unsupported"] = removed
        out["uncaptured"] = uncaptured_words(text, out)
    out["plant"] = plant_name
    out["cause"] = "unknown"  # enforced in code: we never record a cause from a note
    return out


def plain_observation(plant_name, text):
    """The reading used when Gemma is switched off: the gardener's exact words, nothing added."""
    return {"plant": plant_name, "noticed": [text], "conditions": [], "comparison": "",
            "parsed_by": "no_model", "uncaptured": [], "removed_unsupported": [], "cause": "unknown"}


def plain_look_first():
    """The Look First question used when Gemma is switched off. Always asks the gardener to look first."""
    return {"decision": "LOOK_FIRST", "question": FALLBACK_QUESTION,
            "reason": "Look before changing anything.", "source": "builtin"}


def look_first(model, plant_name, plant, ask=ask_gemma):
    """Look First gate. Code, not the model, decides when evidence is enough.

    This is a basic gate, not proof that the evidence is scientifically sufficient. Trials
    unlock once the gardener has saved a follow-up. Gemma can only end the questions early
    when there are at least two notes and one follow-up.
    """
    notes = [o for o in plant["observations"] if o["kind"] == "observation"]
    followups = [o for o in plant["observations"] if o["kind"] == "follow-up"]
    out = ask(model, LOOK_SYS, f"Plant: {plant_name}\nHistory:\n{history_text(plant)}")
    question, reason, source = FALLBACK_QUESTION, "Look before changing anything.", "fallback"
    decision = "LOOK_FIRST"
    if isinstance(out, dict) and isinstance(out.get("question"), str) and out["question"].strip():
        question = out["question"].strip()[:300]
        reason = str(out.get("reason", "")).strip()[:300] or reason
        source = "gemma"
        if OFF_TOPIC_RE.search(question) and not OFF_TOPIC_RE.search(history_text(plant)):
            question, reason, source = FALLBACK_QUESTION, "Look before changing anything.", "guard"
        if out.get("decision") == "ENOUGH_EVIDENCE" and len(notes) >= 2 and len(followups) >= 1:
            decision = "ENOUGH_EVIDENCE"
    return {"decision": decision, "question": question, "reason": reason, "source": source}


def propose_trial(model, plant_name, plant, ask=ask_gemma):
    """Gemma picks from a fixed menu; code builds the trial. Keeps a 1B model safe.

    Gemma's free text is never shown to the gardener here, so it cannot slip in a diagnosis.
    """
    wait = {"trial_recommended": False, "note": WAIT_NOTE, "rejected": False}
    allowed = allowed_options(plant_name)
    out = ask(model, trial_sys(allowed), f"Plant: {plant_name}\nHistory:\n{history_text(plant, 10)}")
    if not isinstance(out, dict):
        return dict(wait, rejected=True)
    choice = str(out.get("choice", "")).strip().lower()
    if choice == KEEP_WATCHING:
        return wait
    if choice not in allowed:  # off the menu, or not suitable for this plant
        return dict(wait, rejected=True)
    measure = str(out.get("measure", "")).strip()[:80] or "open flowers"
    return build_trial(choice, measure, plant, chosen_by="Gemma")


def build_trial(choice, measure, plant, chosen_by="you"):
    """Turn a menu choice into a trial. Used for Gemma's pick and for the gardener's own pick."""
    opt = TRIAL_OPTIONS[choice]
    ordered = sorted(plant["observations"], key=lambda o: o.get("date", ""))
    latest = ordered[-1]["text"][:140] if ordered else ""
    return {
        "trial_recommended": True,
        "question": f"Does this change anything for this plant: {opt['change'].lower()}?",
        "hypothesis": opt["hypothesis"],
        "change": opt["change"],
        "keep_same": opt["keep_same"],
        "comparison": COMPARISON_TEXT,
        "measure": (measure or "").strip()[:80] or "open flowers",
        "duration_days": TRIAL_DAYS,
        "chosen_by": chosen_by,
        "based_on": latest if chosen_by == "Gemma" else "",
        "note": "",
    }


def repeat_trial(t, start=None):
    """A fresh, empty trial of the same change. Never merges with the old one, so each run stands alone."""
    keep = ("question", "hypothesis", "change", "keep_same", "comparison", "measure", "duration_days")
    new = {k: t[k] for k in keep if k in t}
    new.update({"measure2": t.get("measure2", ""), "start": start or today(), "status": "active",
                "records": [], "has_control": t.get("has_control", True), "start_changes": [],
                "chosen_by": t.get("chosen_by", "you"), "repeat_of": t.get("start", "")})
    return new


# ---------- guard events (only things the app actually detected) ----------
def events_from_extract(structured):
    kinds = []
    if structured.get("parsed_by") == "fallback" and structured.get("fallback_code") in EVENT_LABELS:
        kinds.append(structured["fallback_code"])
    if structured.get("removed_unsupported"):
        kinds.append("unsupported_detail_removed")
    return kinds


def event_from_look(result):
    if result.get("source") == "fallback":
        return "builtin_question_used"
    if result.get("source") == "guard":
        return "offtopic_question_blocked"
    return None


def event_from_trial(proposal):
    return "trial_pick_rejected" if proposal.get("rejected") else None


def log_event(data, kind, plant=""):
    if kind not in EVENT_LABELS:
        return
    data.setdefault("guard_events", []).append({"date": today(), "kind": kind, "plant": plant})


def guard_summary(data):
    """Counts of detected events, as {label: count}, only for events that happened."""
    counts = {}
    for e in data.get("guard_events", []):
        if e.get("kind") in EVENT_LABELS:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return {EVENT_LABELS[k]: counts[k] for k in EVENT_LABELS if k in counts}


# ---------- photos ----------
def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "plant"


def save_photo(plant_name, date_iso, filename, content):
    """Validate, shrink and store an uploaded photo; return its path relative to the data folder.

    The photo is re-saved as a JPEG no larger than 1600 px, which also drops EXIF data
    such as GPS location, so photos are safe to commit or share.
    """
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext not in PHOTO_TYPES:
        raise ValueError("Photos must be JPG, PNG or WEBP. (iPhone HEIC photos need converting first.)")
    if not content:
        raise ValueError("That photo is empty.")
    if len(content) > MAX_PHOTO_BYTES:
        raise ValueError("That photo is over 25 MB. Resize it and try again.")
    try:
        img = Image.open(io.BytesIO(content))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
    except Exception:
        raise ValueError("That file is not a readable image.")
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{_slug(plant_name)}-{date_iso}"
    n = 1
    while (PHOTO_DIR / f"{base}-{n}.jpg").exists():
        n += 1
    path = PHOTO_DIR / f"{base}-{n}.jpg"
    img.save(path, "JPEG", quality=88)
    return f"photos/{path.name}"


def photo_path(rel):
    """Return the real path of a saved photo, or None if it is missing."""
    if not rel:
        return None
    p = PHOTO_DIR / Path(str(rel)).name
    return p if p.is_file() else None


# ---------- results ----------
def distinct_days(records):
    return len({r["date"] for r in records})


def last_counts(records):
    """Most recent saved counts, used as the starting values of the next entry."""
    if not records:
        return {k: 0 for k in COUNT_KEYS}
    r = max(records, key=lambda r: r["date"])
    return {k: _count(r.get(k, 0)) for k in COUNT_KEYS}


def backfilled(records):
    """Dates that were entered on a later day than the date they describe."""
    rs = sorted(records, key=lambda r: r["date"])
    return [r["date"] for r in rs if r.get("logged_on") and r["logged_on"] != r["date"]]


def trial_changes(t):
    """Keys of 'what else changed' flags reported at the start or on any entry."""
    flags = set(t.get("start_changes") or [])
    for r in t.get("records", []):
        flags.update(r.get("changes") or [])
    return [k for k in CHANGE_FLAGS if k in flags]


def table_rows(records, has_control, measure="count", measure2=""):
    bf = set(backfilled(records))
    any_changes = any(r.get("changes") for r in records)
    any_notes = any(r.get("note") for r in records)
    rows = []
    for r in sorted(records, key=lambda r: r["date"]):
        row = {"date": r["date"]}
        if has_control:
            row[f"{measure} (changed plant)"] = r.get("changed", 0)
            row[f"{measure} (comparison plant)"] = r.get("control", 0)
            if measure2:
                row[f"{measure2} (changed plant)"] = r.get("changed2", 0)
                row[f"{measure2} (comparison plant)"] = r.get("control2", 0)
        else:
            row[measure] = r.get("changed", 0)
            if measure2:
                row[measure2] = r.get("changed2", 0)
        if bf:
            row["entry"] = "backfilled" if r["date"] in bf else "same day"
        if any_changes:
            row["also changed"] = ", ".join(CHANGE_FLAGS[k] for k in CHANGE_FLAGS if k in (r.get("changes") or [])) or "-"
        if any_notes:
            row["note"] = r.get("note") or "-"
        rows.append(row)
    return rows


def series_labels(t):
    """Which count fields a trial records, and what to call each one: {field: label}."""
    m1, m2 = t.get("measure") or "count", t.get("measure2", "")
    hc = bool(t.get("has_control", True)) and not t.get("observe_only")
    labels = {"changed": f"{m1} (changed plant)" if hc else m1}
    if hc:
        labels["control"] = f"{m1} (comparison plant)"
    if m2:
        labels["changed2"] = f"{m2} (changed plant)" if hc else m2
        if hc:
            labels["control2"] = f"{m2} (comparison plant)"
    return labels


def chart_series(t):
    """Counts by date for a line chart: {"index": [dates], "series": {label: [values]}}. Empty if under two days."""
    rs = sorted(t.get("records", []), key=lambda r: r["date"])
    if distinct_days(rs) < 2:
        return {}
    return {"index": [r["date"] for r in rs],
            "series": {label: [r.get(key, 0) for r in rs] for key, label in series_labels(t).items()}}


JUMP_MIN = 5


def big_jumps(last, current, labels):
    """Counts that moved a lot since the last entry, as readable strings. Used to ask 'are you sure?' once."""
    out = []
    for key, label in labels.items():
        was, now = int(last.get(key, 0)), int(current.get(key, 0))
        diff = abs(now - was)
        if diff >= JUMP_MIN and diff >= was:
            out.append(f"{label}: {was} to {now}")
    return out


def latest_photo(t):
    """(date, path) of the newest saved photo in a trial, or None. Shown so the next photo can match its angle."""
    for r in sorted(t.get("records", []), key=lambda r: r.get("date", ""), reverse=True):
        path = photo_path(r.get("photo"))
        if path:
            return r.get("date", ""), path
    return None


def note_is_late(o):
    """True if a note was written on a later day than the day it describes."""
    return bool(o.get("logged_on")) and o["logged_on"] != o.get("date")


def _compare(rs, ckey, kkey, has_control):
    changed = rs[-1].get(ckey, 0) - rs[0].get(ckey, 0)
    if not has_control:
        if abs(changed) < 2:
            return "Inconclusive: the count barely moved. With no comparison plant this is weak evidence."
        word = "rose" if changed > 0 else "fell"
        return (f"The count {word} by {abs(changed)}, but with no comparison plant "
                "I can't say the change caused it. Weak, before/after evidence only.")
    control = rs[-1].get(kkey, 0) - rs[0].get(kkey, 0)
    gap = changed - control
    if abs(gap) < 2:
        return "Inconclusive: the changed plant and the comparison moved about the same."
    if gap > 0:
        return "Promising, but not conclusive: the changed plant gained more than the comparison. One short trial can't prove a cause."
    return "Opposite of the hypothesis: the comparison gained more. Worth repeating before drawing any conclusion."


def verdict(records, has_control=True, observe_only=False, measure="count", measure2=""):
    """First record (sorted by date) is the baseline. Needs two different days."""
    rs = sorted(records, key=lambda r: r["date"])
    if distinct_days(rs) < 2:
        return "Not enough data yet. Add a baseline and a measurement from a later day."
    if observe_only:
        counts = f"the count went from {rs[0]['changed']} to {rs[-1]['changed']}"
        if measure2:
            counts = (f"{measure} went from {rs[0]['changed']} to {rs[-1]['changed']}; "
                      f"{measure2} went from {rs[0].get('changed2', 0)} to {rs[-1].get('changed2', 0)}")
        return (f"Observation only: {counts}. No change was made during these counts, so this shows what the "
                "plant did while you watched and says nothing about a cause.")
    first = _compare(rs, "changed", "control", has_control)
    if measure2:
        return f"{measure}: {first}\n\n{measure2}: {_compare(rs, 'changed2', 'control2', has_control)}"
    return first


def _confounded(t):
    """True if other changes make it impossible to tie the result to the one tested change."""
    flags = trial_changes(t)
    if not flags or t.get("observe_only"):
        return False
    if any(f not in SHARED_FLAGS for f in flags):
        return True
    return not t.get("has_control", True)  # weather alone only matters with no comparison plant


def _direction(t):
    """+1 / -1 / 0: which way the first measure moved (against the comparison plant if there is one)."""
    rs = sorted(t.get("records", []), key=lambda r: r["date"])
    if distinct_days(rs) < 2:
        return 0
    delta = rs[-1].get("changed", 0) - rs[0].get("changed", 0)
    if t.get("has_control", True):
        delta -= rs[-1].get("control", 0) - rs[0].get("control", 0)
    return 0 if abs(delta) < 2 else (1 if delta > 0 else -1)


def evidence_level(t, siblings=None):
    """How strong the evidence is, 0 to 3. Calculated here, never by Gemma, and never a proof of cause.

    0 observation only (or other changes make it impossible to tell)
    1 before and after, no comparison plant
    2 a comparison plant, no other changes
    3 two finished trials of the same change, each with a comparison plant and no other
      changes, all pointing the same way
    """
    records = t.get("records", [])
    has_control = t.get("has_control", True)
    missing, notes = [], []
    name = None
    flags = trial_changes(t)

    if t.get("observe_only"):
        level = 0
        missing.append("No change was made, so there is nothing to test yet.")
    elif distinct_days(records) < 2:
        level = 0
        missing.append("Measurements from at least two different days.")
    else:
        level = 2 if has_control else 1
        if not has_control:
            missing.append("A comparison plant.")
        if flags:
            labels = "; ".join(CHANGE_FLAGS[k] for k in flags)
            notes.append(f"You reported: {labels}.")
            if _confounded(t):
                level = 0
                name = "Cannot separate the change"
                missing.append("Other things changed too, so the result cannot be tied to your one change.")
            else:
                notes.append("Weather reaches both plants alike, so it is recorded but does not lower the level.")
        if level >= 1 and t.get("status") == "done":
            group = [x for x in (siblings or []) if x.get("status") == "done" and not x.get("observe_only")
                     and x.get("change") == t.get("change")]
            if (any(x is t for x in group) and len(group) >= 2
                    and all(x.get("has_control", True) and not _confounded(x) for x in group)
                    and len({_direction(x) for x in group}) == 1 and _direction(group[0]) != 0):
                level = 3
        if level < 3:
            missing.append("Not repeated yet." if level > 0 else "A repeat once the other changes are removed.")
    if backfilled(records):
        notes.append("Some entries were logged after the date they describe (backfilled): "
                     + ", ".join(backfilled(records)) + ".")
    return {"level": level, "name": name or LEVEL_NAMES[level], "missing": missing, "notes": notes}


def trial_verdict(t, siblings=None):
    base = verdict(t["records"], t.get("has_control", True), t.get("observe_only", False),
                   t.get("measure") or "count", t.get("measure2", ""))
    flags = trial_changes(t)
    if flags and not t.get("observe_only") and distinct_days(t["records"]) >= 2:
        labels = "; ".join(CHANGE_FLAGS[k].lower() for k in flags)
        if _confounded(t):
            base += f"\n\nYou also reported: {labels}. That cannot be separated from your change, so treat this result as inconclusive."
        else:
            base += f"\n\nYou also reported: {labels}. It should have reached both plants, but it is noted."
    return base


# ---------- what is due today, and what to do next ----------
def _parse_day(text):
    try:
        return dt.date.fromisoformat(str(text))
    except ValueError:
        return None


def _count_due_text(t, on):
    m1, m2 = t.get("measure") or "count", t.get("measure2", "")
    measures = m1 + (f" and {m2}" if m2 else "")
    records = sorted(t.get("records", []), key=lambda r: r.get("date", ""))
    if not records:
        return ("Enter your first count." if t.get("observe_only")
                else "Enter your first count BEFORE you change anything.")
    last = records[-1]
    if last.get("date") == on.isoformat():
        return "Counted today. Nothing more to do."
    first_day = _parse_day(records[0].get("date"))
    duration = t.get("duration_days", TRIAL_DAYS)
    text = ""
    if first_day:
        day = max((on - first_day).days, 0)
        text = f"Day {day} of {duration}: "
    text += f"count {measures}, same spot and angle"
    if t.get("has_control", True) and not t.get("observe_only"):
        text += ", and the comparison plant too"
    text += f". Last count: {last.get('changed', 0)} on {last.get('date', '')}."
    if first_day and (on - first_day).days >= duration and distinct_days(records) >= 2:
        text += " You have enough days logged to finish this trial."
    return text


def today_items(data, on=None):
    """A short list of what is due: questions to go and look at, and counts to take. Nothing else."""
    on = on or dt.date.today()
    items = []
    for name in sorted(data.get("plants", {})):
        plant = data["plants"][name]
        pending = plant.get("pending")
        if pending and pending.get("question"):
            items.append({"plant": name, "kind": "look", "text": f"Go and look: {pending['question']}"})
        for t in plant.get("trials", []):
            if t.get("status") == "active":
                items.append({"plant": name, "kind": "count", "text": _count_due_text(t, on)})
    return items


def next_step(t, siblings=None):
    """One plain sentence on what would make the evidence stronger. Chosen in code, never by Gemma."""
    if distinct_days(t.get("records", [])) < 2:
        return "Log counts on at least two different days."
    if t.get("observe_only"):
        return ("You watched without changing anything. To test an idea, pick one change "
                "and keep a similar plant unchanged.")
    ev = evidence_level(t, siblings)
    if ev["name"] == "Cannot separate the change":
        return "Repeat the trial later, with nothing else changing."
    if ev["level"] <= 1:
        return "Repeat with a similar plant left unchanged, so you have something to compare with."
    if ev["level"] == 2:
        return "Repeat the same change once more to see whether the result holds."
    return "Your repeated trials agree. That is still evidence, not proof of a cause."


# ---------- illustrative sample garden ----------
def sample_garden():
    """A made-up garden so anyone can try the full loop. Never written to garden.json."""
    d = dt.date.today()

    def day(n):
        return (d - dt.timedelta(days=n)).isoformat()

    def structured(plant, noticed, conditions):
        return {"plant": plant, "noticed": noticed, "conditions": conditions, "comparison": "",
                "parsed_by": "sample", "uncaptured": [], "removed_unsupported": [], "cause": "unknown"}

    basil_name = "Sample basil (illustrative)"
    basil = get_plant({"plants": {}}, basil_name)
    basil["sample"] = True
    basil["observations"] = [
        {"date": day(8), "kind": "observation", "structured": structured(
            basil_name, ["new leaves seem slow to appear", "2 yellow leaves at the bottom"], ["shady windowsill"]),
         "text": "[Sample] New leaves seem slow to appear. 2 yellow leaves at the bottom. The pot sits on a shady windowsill."},
        {"date": day(7), "kind": "follow-up", "question_source": "sample",
         "question": "Are the newest leaves at the top folded and tiny?",
         "text": "[Sample] Looked at the top: the newest leaves are folded and about 3 mm long."},
    ]
    basil["ready_for_trial"] = True
    trial = build_trial("brighter", "new leaves", basil, chosen_by="you")
    trial.update({"measure2": "", "start": day(4), "status": "done", "has_control": True,
                  "surprise": "[Sample] The comparison basil also grew a leaf after the rainy day.",
                  "records": [
                      {"date": day(4), "changed": 2, "control": 2, "logged_on": day(4)},
                      {"date": day(2), "changed": 3, "control": 2, "logged_on": day(2), "changes": ["weather"],
                       "note": "[Sample] Rained in the afternoon."},
                      {"date": day(0), "changed": 5, "control": 3, "logged_on": day(0)},
                  ]})
    basil["trials"] = [trial]

    mint_name = "Sample mint (illustrative)"
    mint = get_plant({"plants": {}}, mint_name)
    mint["sample"] = True
    mint["observations"] = [
        {"date": day(0), "kind": "observation", "structured": structured(
            mint_name, ["stems look leggy", "leaves are small"], []),
         "text": "[Sample] Mint stems look leggy and the leaves are small."},
    ]
    mint["pending"] = {"decision": "LOOK_FIRST",
                       "question": "Are the new leaves closer together near the top of the stem than lower down?",
                       "reason": "Look at how the newest growth is spaced before changing anything.",
                       "source": "sample"}
    return {"plants": {basil_name: basil, mint_name: mint}}


def summary_markdown(name, plant):
    """A plain-text write-up of one plant, handy for notes or a blog post."""
    out = [f"# {name}: garden notes", ""]
    if plant.get("facts"):
        out += [f"Details the gardener gave: {' '.join(str(plant['facts']).split())}", ""]
    out += ["## Observations"]
    for o in plant["observations"]:
        later = ", logged later" if note_is_late(o) else ""
        out.append(f"- {o['date']} ({o['kind']}{later}): {' '.join(str(o['text']).split())}")
    if not plant["observations"]:
        out.append("- none yet")
    for t in plant["trials"]:
        m1, m2 = t.get("measure") or "count", t.get("measure2", "")
        counting = m1 + (f" and {m2}" if m2 else "")
        out += ["", f"## Trial ({t['status']}): {t['question']}",
                f"Counting: {counting}. Change made: {t.get('change', 'n/a')}.", ""]
        if t["records"]:
            rows = table_rows(t["records"], t.get("has_control", True), m1, m2)
            heads = list(rows[0])
            out.append("| " + " | ".join(h.replace("|", "/") for h in heads) + " |")
            out.append("|" + "---|" * len(heads))
            for row in rows:
                out.append("| " + " | ".join(str(row[h]).replace("|", "/") for h in heads) + " |")
            ev = evidence_level(t, plant["trials"])
            out += ["", trial_verdict(t, plant["trials"]), "",
                    f"Evidence level {ev['level']} of 3: {ev['name']}."]
            out += [f"Next step: {next_step(t, plant['trials'])}"]
            out += [f"Missing: {m}" for m in ev["missing"]]
            out += ev["notes"]
        else:
            out.append("No measurements yet.")
        if t.get("surprise"):
            out += ["", f"What surprised me: {t['surprise']}"]
    out += ["", "Notes and counts are the gardener's own. A short trial cannot prove a cause."]
    return "\n".join(out) + "\n"