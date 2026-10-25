"""
Extract MCQs (text + diagrams) from every PDF in this folder with local models (Ollama).

    python Extract.py

Two passes, so each model gets the whole GPU in turn (RTX 3050 + 32 GB RAM friendly):
  PASS 1  vision model reads each page -> question text, options, answer key (if printed),
          and bounding boxes of any diagrams in the question or in the options.
          Diagrams are cropped from the page and saved to ./images/.
  PASS 2  a separate text *reasoning* model rates difficulty (Easy/Medium/Hard) using a
          JEE rubric, instead of the vision model guessing.

Resumable: progress is kept in extraction_state.json, so after a crash or Ctrl+C just run it
again. Final output: extracted_database.json (same fields as before + image paths).

Setup:
    pip install ollama pdf2image pillow       (and install poppler)
    ollama pull qwen3-vl:4b                   # vision/OCR   (~6 GB; use qwen3-vl:4b on a 6 GB card)
    ollama pull qwen3:8b                      # difficulty   (~5 GB; qwen3:14b is smarter but ~2-3x slower here)
"""
import base64
import json
import re
from io import BytesIO
from pathlib import Path
import hashlib
import ollama
from pdf2image import convert_from_path
from PIL import ImageDraw

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "extracted_database.json"
STATE_FILE = HERE / "extraction_state.json"
FAILED_LOG = HERE / "failed_pages.log"
IMG_DIR = HERE / "images"
DEBUG_DIR = HERE / "debug_boxes"

# ---------------------------------------------------------------- configuration
VISION_MODEL = "qwen3-vl:4b"   # good OCR + grounding (bounding boxes). Try "qwen3-vl:4b" if VRAM is tight.
REASON_MODEL = "qwen3:8b"      # text-only model used for difficulty rating
DPI = 200
SEND_MAX_SIDE = 1568           # page is downscaled to this before going to the model; crops use the full render
BOX_SCALE = "norm1000"         # qwen3-vl returns boxes on a 0-1000 grid. For qwen2.5vl use "pixels".
DIFFICULTY_VOTES = 1           # 3 = more stable ratings (median), but ~3x slower
DEBUG_BOXES = True             # save pages with the detected boxes drawn, so you can eyeball them

SUBJECTS = ["Physics", "Chemistry", "Math"]
LEVELS = ["Easy", "Medium", "Hard"]
OPTS = ["a", "b", "c", "d"]

# ---------------------------------------------------------------- pass 1: schema + prompt
_str = {"type": "string"}
_box = {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 4}

_props = {
    "subject": {"type": "string", "enum": SUBJECTS},
    "chapter": _str,
    "question_text": _str,
    **{f"option_{o}": _str for o in OPTS},
    "correct_option": {"type": "string", "enum": ["A", "B", "C", "D", "NONE"]},
    "question_figure": _box,
    **{f"option_{o}_figure": _box for o in OPTS},
}
SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "object", "properties": _props, "required": list(_props)},
        }
    },
    "required": ["questions"],
}

PROMPT = r"""
You are a careful JEE data-entry assistant. Look at this page of a question paper.
Extract every multiple-choice question that is fully visible on the page.
- Copy the text exactly. Do NOT solve the questions.
- Ignore watermarks, page numbers and headers.
- Write all mathematics as LaTeX inside \( ... \), for example \( x^2 + y^2 = r^2 \).
- subject must be Physics, Chemistry or Math.
- chapter is the topic the question belongs to.
- correct_option is A, B, C or D ONLY if the page itself shows the answer (answer key or
  worked solution). Otherwise use NONE. Never guess the answer.

DIAGRAMS
- question_figure: if the question contains a diagram, graph, circuit, apparatus or molecule
  picture, give its bounding box as [x_min, y_min, x_max, y_max]. The box must contain ONLY the
  picture, not the question text. If there is no picture use [].
- option_a_figure ... option_d_figure: if an option is itself a picture (e.g. four graphs or
  four structures), give the bounding box of just that option's picture, and put any printed label/text
  in the option text (use "" if none). If the option is plain text use [].
- Coordinates: __COORDS__

If the page contains no questions, return an empty list.
Return JSON shaped like {"questions": [ {...}, ... ]}.
"""


def coords_hint(w, h):
    if BOX_SCALE == "norm1000":
        return ("integers from 0 to 1000 relative to the image (0,0 = top-left, 1000,1000 = bottom-right), "
                "independent of the image's pixel size.")
    return f"pixel coordinates in this {w}x{h} image, (0,0) = top-left."


# ---------------------------------------------------------------- pass 2: schema + prompt
DIFF_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": _str,
        "steps": {"type": "integer"},
        "complexity": {"type": "string", "enum": LEVELS},
    },
    "required": ["reasoning", "steps", "complexity"],
}

DIFF_PROMPT = """You are a senior JEE Main/Advanced paper-setter rating question difficulty for a
student with a good preparation level.

Rubric:
- Easy:   one concept or formula applied directly, or pure recall (definitions, units, standard
          values, simple substitution). 1-2 steps, little calculation.
- Medium: two or three steps, or a standard problem type that needs a couple of concepts combined,
          moderate algebra/calculus/arithmetic.
- Hard:   multi-concept, 4+ steps, lengthy or tricky calculation, non-obvious insight, easily
          confused options, or a conceptual trap.
Typical JEE papers are roughly 30% Easy, 45% Medium, 25% Hard - do not default to Medium.

First write at most two sentences of reasoning (what is needed to solve it), then the number of
steps, then the label. Do not output the final numeric answer.

Subject: {subject}
Chapter: {chapter}
Question: {question}
(A) {a}
(B) {b}
(C) {c}
(D) {d}
{figure_note}"""


# ---------------------------------------------------------------- helpers
def fix_latex_escapes(s: str) -> str:
    """LaTeX uses single backslashes, which are invalid JSON (\\( \\alpha) or silently
    become control characters (\\frac -> form feed, \\theta -> tab). Keep existing \\\\ and
    \\" pairs, double every other lone backslash."""
    return re.sub(r'\\\\|\\"|\\', lambda m: m.group(0) if len(m.group(0)) == 2 else "\\\\", s)


def parse_json_list(raw: str) -> list:
    m = re.search(r"\{.*\}|\[.*\]", raw, re.S)  # ignore prose or ``` fences around the JSON
    if not m:
        raise ValueError("no JSON found in model output")
    data = json.loads(fix_latex_escapes(m.group(0)))
    if isinstance(data, dict):
        data = data.get("questions", [data])
    if not isinstance(data, list):
        raise ValueError("JSON is not a list of questions")
    return data


def normalize_subject(s):
    s = str(s or "").strip().lower()
    if s.startswith("phys"):
        return "Physics"
    if s.startswith("chem"):
        return "Chemistry"
    if s.startswith("math"):
        return "Math"
    return None


def to_jpeg_b64(image, max_side):
    img = image.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), img.size


def box_to_pixels(box, page_w, page_h, sent_w, sent_h):
    """Model box -> validated pixel box on the full-resolution page, or None."""
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if BOX_SCALE == "norm1000":
        fx, fy = page_w / 1000, page_h / 1000
    else:
        fx, fy = page_w / sent_w, page_h / sent_h
    x0, x1 = sorted((x0 * fx, x1 * fx))
    y0, y1 = sorted((y0 * fy, y1 * fy))
    pad = 0.006 * max(page_w, page_h)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(page_w, x1 + pad), min(page_h, y1 + pad)
    w, h = x1 - x0, y1 - y0
    if w < 0.03 * page_w or h < 0.02 * page_h:      # too small to be a real figure
        return None
    if w * h > 0.8 * page_w * page_h:                # model boxed the whole page
        return None
    return (int(x0), int(y0), int(x1), int(y1))


def save_crop(page, box, name):
    IMG_DIR.mkdir(exist_ok=True)
    path = IMG_DIR / name
    page.crop(box).save(path, format="PNG")
    return f"images/{name}"


def clean(item, page, pdf_stem, page_no, idx, boxes_log):
    """Validated, normalized question dict (with cropped figure paths), or None if unusable."""
    if not isinstance(item, dict):
        return None
    subject = normalize_subject(item.get("subject"))
    q_text = str(item.get("question_text") or "").strip()
    if not subject or not q_text:
        return None

    page_w, page_h = page.size
    sent_w, sent_h = item.pop("_sent_size")
    base = f"{pdf_stem}_p{page_no:03d}_q{idx:02d}"

    def figure(key, suffix):
        box = box_to_pixels(item.get(key), page_w, page_h, sent_w, sent_h)
        if not box:
            return None
        boxes_log.append((box, suffix))
        return save_crop(page, box, f"{base}_{suffix}.png")

    out = {
        "subject": subject,
        "chapter": str(item.get("chapter") or "").strip() or "Uncategorised",
        "question_text": q_text,
        "question_image": figure("question_figure", "question"),
    }
    for o in OPTS:
        img = figure(f"option_{o}_figure", f"opt{o.upper()}")
        text = str(item.get(f"option_{o}") or "").strip()
        if not text and not img:
            return None  # an option with neither text nor picture = bad extraction
        out[f"option_{o}"] = text or "(see image)"
        out[f"option_{o}_image"] = img

    key = str(item.get("correct_option") or "").strip().upper()[:1]
    out["correct_option"] = key if key in ("A", "B", "C", "D") else None
    out["complexity"] = None  # filled in pass 2
    return out


# ---------------------------------------------------------------- pass 1
def extract_page(page, label, pdf_stem, page_no):
    img_b64, (sw, sh) = to_jpeg_b64(page, SEND_MAX_SIDE)
    prompt = PROMPT.replace("__COORDS__", coords_hint(sw, sh))
    messages = [{"role": "user", "content": prompt, "images": [img_b64]}]
    opts = {"temperature": 0, "num_ctx": 8192}

    try:
        resp = ollama.chat(model=VISION_MODEL, messages=messages, format=SCHEMA, options=opts)
    except Exception as e:  # some Ollama builds reject schema + images; retry unconstrained
        print(f"  structured output failed ({e}); retrying without schema")
        resp = ollama.chat(model=VISION_MODEL, messages=messages, options=opts)

    raw = resp["message"]["content"]
    try:
        items = parse_json_list(raw)
    except Exception as e:
        print(f"  could not parse this page: {e} (raw output saved to {FAILED_LOG.name})")
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(f"--- {label} ---\n{raw}\n\n")
        return []

    boxes_log, cleaned = [], []
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            item["_sent_size"] = (sw, sh)
        c = clean(item, page, pdf_stem, page_no, i, boxes_log)
        if c:
            cleaned.append(c)
    if len(cleaned) < len(items):
        print(f"  dropped {len(items) - len(cleaned)} incomplete question(s)")

    n_fig = sum(bool(q["question_image"]) + sum(bool(q[f"option_{o}_image"]) for o in OPTS) for q in cleaned)
    if n_fig:
        print(f"  saved {n_fig} figure crop(s)")
    if DEBUG_BOXES and boxes_log:
        DEBUG_DIR.mkdir(exist_ok=True)
        dbg = page.convert("RGB").copy()
        draw = ImageDraw.Draw(dbg)
        for box, tag in boxes_log:
            draw.rectangle(box, outline="red", width=5)
            draw.text((box[0] + 8, box[1] + 8), tag, fill="red")
        dbg.save(DEBUG_DIR / f"{pdf_stem}_p{page_no:03d}.jpg", quality=80)
    return cleaned


# ---------------------------------------------------------------- pass 2
def rate_difficulty(q):
    figure_note = ""
    if q["question_image"] or any(q[f"option_{o}_image"] for o in OPTS):
        figure_note = "(The question includes a diagram that you cannot see; judge from the text.)"
    prompt = DIFF_PROMPT.format(
        subject=q["subject"], chapter=q["chapter"], question=q["question_text"],
        a=q["option_a"], b=q["option_b"], c=q["option_c"], d=q["option_d"],
        figure_note=figure_note,
    )
    messages = [{"role": "user", "content": prompt}]
    votes = []
    for _ in range(DIFFICULTY_VOTES):
        opts = {"temperature": 0 if DIFFICULTY_VOTES == 1 else 0.4, "num_ctx": 4096, "num_predict": 400}
        try:
            try:
                resp = ollama.chat(model=REASON_MODEL, messages=messages, format=DIFF_SCHEMA,
                                   options=opts, think=False)
            except TypeError:  # older ollama python client without the `think` argument
                resp = ollama.chat(model=REASON_MODEL, messages=messages, format=DIFF_SCHEMA, options=opts)
            level = json.loads(resp["message"]["content"]).get("complexity")
            if level in LEVELS:
                votes.append(LEVELS.index(level))
        except Exception as e:
            print(f"  difficulty call failed: {e}")
    if not votes:
        return "Medium"
    return LEVELS[sorted(votes)[len(votes) // 2]]  # median


def unload(model):
    try:
        ollama.chat(model=model, messages=[], keep_alive=0)  # free VRAM before the next model loads
    except Exception:
        pass


# ---------------------------------------------------------------- state / main
def load_state():
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = {"pages_done": [], "questions": []}
    state.setdefault("pdfs_done", {})   # {hash: filename}
    return state


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def write_output(state):
    OUT_FILE.write_text(json.dumps(state["questions"], indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    pdfs = sorted(HERE.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {HERE}. Put a PDF next to Extract.py.")
    state = load_state()

    # ---- PASS 1: vision extraction
    new_pdfs = 0
    for pdf in pdfs:
        h = file_hash(pdf)
        if h in state["pdfs_done"]:
            continue                      # already fully processed, don't even open it
        # Old state files used "name.pdf#page" tags; still honour them
        legacy_done = any(t.startswith(pdf.name + "#") for t in state["pages_done"])
        new_pdfs += 1
        print(f"\n--- NEW: {pdf.name} ---")
        pages = convert_from_path(str(pdf), dpi=DPI)
        stem = f"{pdf.stem}_{h[:6]}"      # unique image names per file
        for i, page in enumerate(pages, start=1):
            tag = f"{h}#{i}"
            if tag in state["pages_done"] or f"{pdf.name}#{i}" in state["pages_done"]:
                continue
            print(f"Page {i}/{len(pages)}...")
            found = extract_page(page, f"{pdf.name} page {i}", stem, i)
            print(f"  {len(found)} question(s)")
            state["questions"].extend(found)
            state["pages_done"].append(tag)
            save_state(state)
            write_output(state)
        state["pdfs_done"][h] = pdf.name  # mark whole PDF finished
        save_state(state)
    unload(VISION_MODEL)
    if new_pdfs == 0:
        print("\nNo new PDFs found.")

    # ---- PASS 2: difficulty with the reasoning model
    todo = [q for q in state["questions"] if not q.get("complexity")]
    if todo:
        print(f"\nRating difficulty of {len(todo)} question(s) with {REASON_MODEL}...")
    for n, q in enumerate(todo, start=1):
        q["complexity"] = rate_difficulty(q)
        print(f"  [{n}/{len(todo)}] {q['subject']} / {q['chapter']}: {q['complexity']}")
        save_state(state)
        write_output(state)
    unload(REASON_MODEL)

    qs = state["questions"]
    keyed = sum(q["correct_option"] is not None for q in qs)
    figs = sum(bool(q["question_image"]) for q in qs)
    print(f"\nDone: {len(qs)} questions ({keyed} with an answer key, {figs} with a question diagram) "
          f"-> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
