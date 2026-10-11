# 🌱 Garden Trials

**Look first. Change one thing. Let the garden answer.**

A small local app for gardeners. A Gemma model running on your own laptop helps you write down what you
see and asks one thing to go and check on the plant. Plain Python builds the experiment, keeps the
records and works out how much the evidence really shows. The app never says what caused a result.

Built for the Hacktoberfest Open-Source AI Challenge, Week 1: Touch Grass.

**Demo video (48 seconds, no sound):** https://youtu.be/Rm7X4btlZVk

## Quickstart

1. Install [Python 3.11 or newer](https://www.python.org/downloads/) and [Ollama](https://ollama.com/download). The tests were run on Python 3.11.
2. Download the code, then in its folder run:

```bash
python -m pip install -r requirements.txt
ollama pull gemma3:1b
python -m streamlit run app.py
```

On Windows you can double-click `run.bat` instead. On macOS or Linux, run `bash run.sh`. Both scripts
still start the app if Ollama is missing, so you can use "Run without Gemma" (see below).

**No Ollama yet?** Tick **"Try the illustrative sample garden"** in the sidebar to walk through the whole loop
with made-up plants. Nothing from the sample garden is ever saved to your real garden. To type your own notes
without Ollama, also tick **"Run without Gemma"**: your words are kept exactly as written, the app asks its
built-in Look First question, and you choose the change yourself. No model is involved in that mode.

**On your phone?** The app runs on your computer. With the phone on the same Wi-Fi, open the _Network URL_
that Streamlit prints when it starts. Keep the computer on.

**See a real garden:** `examples/my-garden/` holds the author's real notes, counts and photos from October 2026.
Open it without touching your own data:

```bash
# macOS / Linux
GARDEN_DIR=examples/my-garden python -m streamlit run app.py
# Windows (Command Prompt)
set GARDEN_DIR=examples\my-garden
python -m streamlit run app.py
# Windows (PowerShell)
$env:GARDEN_DIR="examples\my-garden"; python -m streamlit run app.py
```

## The loop

1. **Observe.** Write a note and say when you saw it. Gemma reads it into plain facts. Anything Gemma adds that you did not write is removed, and if its reading is unusable your exact words are kept.
2. **Look first.** Before any suggestion, the app asks one specific thing to go and check on the plant. Trials stay locked until you write down what you found.
3. **Watch, or change one thing.** Pick one change from a small fixed menu (or only watch), ideally with a similar plant left alone for comparison.
4. **Count, photograph, repeat.** The app shows what is due today, the last photo so you can match its angle, and asks "are you sure?" once if a count jumps a lot.
5. **An honest verdict.** You get the table, a chart, a verdict, an evidence level, and one plain next step.

### Evidence levels (calculated by Python, never by Gemma)

| Level | Meaning                                                                                                              |
| ----- | -------------------------------------------------------------------------------------------------------------------- |
| 0     | Observation only, or other changes make the result impossible to separate from your one change                       |
| 1     | Before and after, no comparison plant                                                                                |
| 2     | With a comparison plant and no other changes                                                                         |
| 3     | Two finished trials of the same change, each with a comparison plant and no other changes, all pointing the same way |

The level shows how strong the evidence is. It never means a cause has been proven. "What else changed?"
(rain, moved plant, watering changed) is recorded at the start and on every entry. Entries logged after
the day they describe are marked as backfilled or logged later.

## What Gemma does, and what Python does

| Gemma (gemma3:1b through Ollama) proposes | Python decides                                                                                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| A reading of your note                    | Removes details you did not write; keeps your exact words if the reading is unusable; never records a cause                                     |
| One question to go and check              | Blocks pest or disease questions you did not raise and asks a built-in question instead; uses the same built-in question if Gemma's is unusable |
| One pick from a fixed menu                | Rejects picks that are off the menu or unsuitable for the plant; builds every trial itself                                                      |
|                                           | Saves your records safely, calculates every verdict and evidence level                                                                          |

The sidebar panel "How Garden Trials handles AI output" lists only the events the app actually detected in
your garden, such as a number Gemma dropped from a note.

## Privacy, precisely

- No cloud AI API is used. Gemma runs on your computer through Ollama.
- Your notes and photos are stored in this folder (`garden.json` and `photos/`). Photos are re-saved as JPEGs, at most 1600 px on the longest side, without their EXIF data, so GPS location is removed.
- Setup needs the internet once, to download the packages, Ollama and the model.
- Streamlit's anonymous usage statistics are switched off in `.streamlit/config.toml`.
- If `garden.json` ever cannot be read, the app moves it aside to `garden.unreadable-<time>.json` and tells you. It never overwrites it.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests replace Gemma with fake and deliberately bad answers (garbage, dropped numbers, invented pests,
off-menu picks). A second set drives the real pages end to end. No Ollama is needed.

## Limits

- A short trial cannot prove a cause. The app says so on every verdict.
- A 1B model can misread notes. The guards catch the failures listed above, not every possible mistake.
- The Look First gate is basic: trials unlock once you have saved one note and one follow-up about what you found. That is not proof the evidence is sufficient.
- It cannot tell whether you really went outside. The pause is something you build into your own habit.
- Checks on Gemma's reading match English (a-z) words. Notes in other scripts, such as Hindi, are not checked the same way.
- Counts and photos are the gardener's own. Different angles and light make photos hard to compare, so use the note to say so.
- The menu is small on purpose. There is no fertilizer, spray or diagnosis advice.
- If `garden.json` is deleted or moved, the app starts with an empty garden and shows no warning. Keep a copy of the file.
- In `examples/my-garden/`, the number check rejected Gemma's reading of all three notes, so the app kept the author's exact words each time.
- Answers can take up to a minute on a laptop CPU. That is the price of running locally.

## Settings

- `OLLAMA_MODEL` (or the sidebar box) changes the model. The default is `gemma3:1b`.
- `OLLAMA_HOST` points the app at an Ollama server on another port or machine, for example `192.168.1.20:11434`. It is the same variable Ollama itself uses.
- `GARDEN_DIR` keeps a separate garden, for example a demo folder.

## License

MIT. See [LICENSE](LICENSE).

## Credits

Built with [Streamlit](https://streamlit.io), [Ollama](https://ollama.com), [Gemma](https://ai.google.dev/gemma),
Pillow, pandas and requests.
The plants, photos, counts and notes are the author's own.
