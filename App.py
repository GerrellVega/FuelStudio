import io
import os
import re
import random
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from PIL import (Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw,
                 ImageFont, ImageChops, ImageFile)

try:  # needs a display; unavailable on headless servers like Render
    from PIL import ImageGrab
except Exception:
    ImageGrab = None

# ImageFile: tolerate slightly truncated/corrupt uploads instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------- helpers ----------
def num(part, default):
    m = re.search(r"-?\d+(?:\.\d+)?", part)
    return float(m.group()) if m else default


def match_size(img, other):
    return other.convert("RGB").resize(img.size)


def get_font(img):
    return ImageFont.load_default(size=max(16, img.width // 12))


# ---------- basic adjustments ----------
def op_grayscale(img, p, raw, ctx, i):
    return ImageOps.grayscale(img).convert("RGB")


def op_brighten(img, p, raw, ctx, i):
    return ImageEnhance.Brightness(img).enhance(num(p, 1.3 + i * 0.1))


def op_darken(img, p, raw, ctx, i):
    return ImageEnhance.Brightness(img).enhance(num(p, 0.75 - i * 0.1))


def op_saturation(img, p, raw, ctx, i):
    return ImageEnhance.Color(img).enhance(num(p, 1.5 + i * 0.3))


def op_contrast(img, p, raw, ctx, i):
    return ImageEnhance.Contrast(img).enhance(num(p, 1.2 + i * 0.1))


# ---------- ImageFilter ----------
def op_blur(img, p, raw, ctx, i):
    return img.filter(ImageFilter.GaussianBlur(num(p, 2 + i * 2)))


def op_sharpen(img, p, raw, ctx, i):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(150 + i * 75)))


def op_edges(img, p, raw, ctx, i):
    return img.filter(ImageFilter.FIND_EDGES)


def op_emboss(img, p, raw, ctx, i):
    return img.filter(ImageFilter.EMBOSS)


def op_contour(img, p, raw, ctx, i):
    return img.filter(ImageFilter.CONTOUR)


def op_smooth(img, p, raw, ctx, i):
    return img.filter(ImageFilter.SMOOTH_MORE)


def op_detail(img, p, raw, ctx, i):
    return img.filter(ImageFilter.DETAIL)


# ---------- ImageDraw / ImageFont ----------
def op_text(img, p, raw, ctx, i):
    m = re.search(r"text[:\s]+(.+)", raw, re.I)
    txt = m.group(1).strip() if m else "FuelStrictStudio"
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = get_font(img)
    y = int(img.height * (0.05 + 0.4 * i))
    d.text((int(img.width * 0.05), y), txt, fill="white", font=font,
           stroke_width=2, stroke_fill="black")
    return img


def op_box(img, p, raw, ctx, i):
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    m = int(min(w, h) * (0.1 + 0.05 * i))
    d.rectangle([m, m, w - m, h - m], outline="red", width=max(3, w // 100))
    return img


def op_watermark(img, p, raw, ctx, i):
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = get_font(img)
    label = "FuelStrictStudio"
    box = d.textbbox((0, 0), label, font=font)
    x = base.width - (box[2] - box[0]) - 20
    y = base.height - (box[3] - box[1]) - 20
    d.text((x, y), label, fill=(255, 255, 255, 140), font=font)
    return Image.alpha_composite(base, layer)


# ---------- ImageChops ----------
def op_difference(img, p, raw, ctx, i):
    return ImageChops.difference(img, match_size(img, ctx["original"]))


def op_multiply(img, p, raw, ctx, i):
    return ImageChops.multiply(img, match_size(img, ctx["original"]))


def op_lighter(img, p, raw, ctx, i):
    return ImageChops.lighter(img, match_size(img, ctx["original"]))


def op_darker(img, p, raw, ctx, i):
    return ImageChops.darker(img, match_size(img, ctx["original"]))


# ---------- Image: rotate .. thumbnail ----------
def op_rotate(img, p, raw, ctx, i):
    if re.search(r"\d", p):
        angle = num(p, 90)
    elif "right" in p or "clockwise" in p:
        angle = -90
    elif "left" in p:
        angle = 90
    else:
        angle = 90 * (i + 1)
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def op_transpose(img, p, raw, ctx, i):
    return img.transpose(Image.Transpose.TRANSPOSE)


def op_crop(img, p, raw, ctx, i):
    pct = min(max(num(p, 80 - i * 10), 10), 100) / 100
    w, h = img.size
    cw, ch = int(w * pct), int(h * pct)
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch))


def op_resize(img, p, raw, ctx, i):
    n = num(p, 50)
    if "%" in p:
        size = (max(1, int(img.width * n / 100)), max(1, int(img.height * n / 100)))
    else:  # treat as target width
        size = (max(1, int(n)), max(1, int(img.height * n / img.width)))
    return img.resize(size, Image.LANCZOS)


def op_thumbnail(img, p, raw, ctx, i):
    out = img.copy()
    n = int(num(p, 256))
    out.thumbnail((n, n))
    return out


# ---------- blend / alpha_composite / paste ----------
def op_blend(img, p, raw, ctx, i):
    other = match_size(img, ctx["overlay"]) if ctx["overlay"] else ImageOps.invert(img)
    return Image.blend(img, other, min(max(num(p, 0.3 + i * 0.2), 0), 1))


def op_composite(img, p, raw, ctx, i):
    base = img.convert("RGBA")
    alpha = int(min(max(num(p, 100 + i * 40), 0), 255))
    if ctx["overlay"]:
        layer = match_size(img, ctx["overlay"]).convert("RGBA")
        layer.putalpha(alpha)
    else:
        colors = [(255, 120, 0), (0, 120, 255), (160, 0, 255)]
        layer = Image.new("RGBA", base.size, colors[i % 3] + (alpha,))
    return Image.alpha_composite(base, layer)


def op_paste(img, p, raw, ctx, i):
    out = img.copy()
    src = (ctx["overlay"] or img).copy()
    src.thumbnail((max(1, out.width // 3), max(1, out.height // 3)))
    m = 20
    corners = [(m, m),
               (out.width - src.width - m, m),
               (m, out.height - src.height - m),
               (out.width - src.width - m, out.height - src.height - m)]
    out.paste(src, corners[i % 4])
    return out


# ---------- ImageOps ----------
def op_autocontrast(img, p, raw, ctx, i):
    return ImageOps.autocontrast(img, cutoff=int(num(p, 2 + i * 2)))


def op_equalize(img, p, raw, ctx, i):
    return ImageOps.equalize(img)


def op_posterize(img, p, raw, ctx, i):
    return ImageOps.posterize(img, int(min(max(num(p, 4 - i), 1), 8)))


def op_mirror(img, p, raw, ctx, i):
    return ImageOps.mirror(img)


def op_flip(img, p, raw, ctx, i):
    if "horizontal" in p:
        return ImageOps.mirror(img)
    return ImageOps.flip(img)


def op_fit(img, p, raw, ctx, i):
    n = int(num(p, 512))
    return ImageOps.fit(img, (n, n))


def op_pad(img, p, raw, ctx, i):
    n = int(num(p, 512))
    return ImageOps.pad(img, (n, n), color="black")


def op_expand(img, p, raw, ctx, i):
    return ImageOps.expand(img, border=int(num(p, 20 + i * 10)), fill="white")


def op_invert(img, p, raw, ctx, i):
    return ImageOps.invert(img)


def op_solarize(img, p, raw, ctx, i):
    return ImageOps.solarize(img, threshold=int(num(p, 128)))


# ---------- ImageGrab ----------
def op_grab(img, p, raw, ctx, i):
    if ImageGrab is None:
        ctx["notes"].append("grab unavailable on this server (no display)")
        return img
    try:
        return ImageGrab.grab().convert("RGB")
    except Exception as e:
        ctx["notes"].append(f"grab failed: {e}")
        return img


# Order matters: first match wins (specific before general)
OPS = [
    (["chops difference", "chops diff"], op_difference),
    (["chops multiply", "multiply"], op_multiply),
    (["chops lighter"], op_lighter),
    (["chops darker"], op_darker),
    (["autocontrast"], op_autocontrast),
    (["grayscale", "greyscale", "grey", "gray", "black and white", "b&w", "monochrome"], op_grayscale),
    (["saturat", "vivid", "colorful"], op_saturation),
    (["darker", "darken", "dim"], op_darken),
    (["bright", "lighten", "lighter"], op_brighten),
    (["contrast"], op_contrast),
    (["blur"], op_blur),
    (["sharp"], op_sharpen),
    (["edge"], op_edges),
    (["emboss"], op_emboss),
    (["contour"], op_contour),
    (["smooth"], op_smooth),
    (["detail"], op_detail),
    (["watermark"], op_watermark),
    (["text"], op_text),
    (["box"], op_box),
    (["rotate"], op_rotate),
    (["transpose"], op_transpose),
    (["crop"], op_crop),
    (["resize"], op_resize),
    (["thumbnail"], op_thumbnail),
    (["composite"], op_composite),
    (["blend"], op_blend),
    (["paste"], op_paste),
    (["equalize"], op_equalize),
    (["posterize"], op_posterize),
    (["mirror"], op_mirror),
    (["flip"], op_flip),
    (["fit"], op_fit),
    (["pad"], op_pad),
    (["expand", "border"], op_expand),
    (["invert"], op_invert),
    (["solarize"], op_solarize),
    (["grab", "screenshot"], op_grab),
]


def apply_commands(base, command, i, overlay=None):
    """Run comma / 'then' / 'and' separated commands in order."""
    ctx = {"original": base,
           "overlay": overlay.convert("RGB") if overlay is not None else None,
           "notes": []}
    img = base.copy()
    matched = False
    for raw in re.split(r",|;|\bthen\b|\s+and\s+", command or "", flags=re.I):
        raw = raw.strip()
        part = raw.lower()
        if not part:
            continue
        for keys, fn in OPS:
            if any(k in part for k in keys):
                img = fn(img, part, raw, ctx, i).convert("RGB")
                matched = True
                break
        else:
            ctx["notes"].append(f"unknown command '{raw}'")
    if not matched:  # default per-slot variation
        if i == 1:
            img = ImageOps.invert(img)
        elif i == 2:
            img = ImageOps.solarize(img, threshold=128)
    return img, ctx["notes"]


# ---------- AI editing (Google Gemini image model) ----------
AI_MODE = "AI edit"
QUICK_MODE = "Quick commands"
MODES = [AI_MODE, QUICK_MODE]
DEFAULT_MODE = AI_MODE if os.environ.get("GEMINI_API_KEY") else QUICK_MODE
GEMINI_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
_client = None


def ai_edit(img, instruction):
    """Send the image + plain-English instruction to Gemini; return the edited PIL image."""
    global _client
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set on the server")
    if not (instruction or "").strip():
        raise RuntimeError("type what you want changed")
    from google import genai
    if _client is None:
        _client = genai.Client(api_key=key)
    src = img.convert("RGB")
    src.thumbnail((1536, 1536))
    prompt = ("Edit this photo as described. Keep everything else the same: same scene, "
              "framing, lighting and style. Edit: " + instruction.strip())
    resp = _client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, src])
    texts = []
    for part in (getattr(resp, "parts", None) or []):
        data = getattr(getattr(part, "inline_data", None), "data", None)
        if data:
            return Image.open(io.BytesIO(data)).convert("RGB")
        if getattr(part, "text", None):
            texts.append(part.text)
    detail = " ".join(texts)[:200] if texts else "The model may have declined this request."
    raise RuntimeError("No image came back. " + detail)


# ---------- session state ----------
MAX_PHOTOS = 10
MAX_SIDE = 2048  # keeps memory low on small servers


def new_state():
    return {"options": [None, None, None], "final": [], "photos": [],
            "slot": None, "orig": None, "work": None, "cmds": [], "undo": [],
            "counter": 0}


def option_images(state):
    return [o["img"] if o else None for o in state["options"]]


def chain(cmds, limit=70):
    text = " → ".join(c for c in cmds if c) or "(no edits)"
    return text if len(text) <= limit else text[:limit - 1] + "…"


def edit_log(state):
    if not state["cmds"]:
        return "_No edits yet._"
    return "\n".join(f"{k + 1}. {c}" for k, c in enumerate(state["cmds"]))


def final_render(state):
    gallery = [(f["img"], f"#{f['id']} · {chain(f['cmds'], 40)}") for f in state["final"]]
    if not state["final"]:
        return gallery, "_Nothing finalized yet._"
    rows = ["| # | Edits | Size |", "|---|---|---|"]
    for f in state["final"]:
        w, h = f["img"].size
        rows.append(f"| {f['id']} | {chain(f['cmds'], 120).replace('|', '/')} | {w}×{h} |")
    return gallery, "\n".join(rows)


def load_photos(files):
    """Open uploaded files (paths or temp-file objects) as upright RGB images."""
    if not files:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    photos = []
    for f in files[:MAX_PHOTOS]:
        path = getattr(f, "name", f)
        try:
            im = Image.open(path)
            im = ImageOps.exif_transpose(im).convert("RGB")  # fix phone-photo rotation
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            photos.append(im)
        except Exception:
            continue
    return photos


def overlay_choices(state):
    return ["None"] + [f"Photo {k + 1}" for k in range(len(state["photos"]))]


def get_overlay(state, choice):
    if choice and choice.startswith("Photo "):
        k = int(choice.split()[1]) - 1
        if 0 <= k < len(state["photos"]):
            return state["photos"][k]
    return None


# ---------- STEP 1: create 1-3 options from one or more photos ----------
def generate(files, command, mode, n, state):
    photos = load_photos(files)
    if not photos:
        return ([state] + option_images(state) + ["Upload at least one photo first.",
                gr.update(), gr.update()])
    n = int(n)
    N = len(photos)
    state["photos"] = photos
    options, notes, errors = [None, None, None], [], []
    cmd = (command or "").strip()

    if mode == AI_MODE:
        if not cmd:
            return ([state] + option_images(state) + ["Describe the edit you want first.",
                    gr.update(), gr.update()])

        def job(i):
            try:
                return ai_edit(photos[i % N], cmd), None
            except Exception as e:
                return None, str(e)[:200]

        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(job, range(n)))
        for i, (out, err) in enumerate(results):
            if out is not None:
                options[i] = {"img": out, "cmds": [f"photo {i % N + 1}: AI: {cmd}"]}
            else:
                errors.append(f"option {i + 1}: {err}")
        if not any(options):  # keep earlier options if everything failed
            return ([state] + option_images(state)
                    + ["AI edit failed. " + " | ".join(errors), gr.update(), gr.update()])
        msg = f"AI created {sum(o is not None for o in options)} option(s) from {N} photo(s)."
        if errors:
            msg += " Failed: " + " | ".join(errors)
    else:
        for i in range(n):
            base = photos[i % N]
            overlay = photos[(i + 1) % N] if N > 1 else None
            out, nt = apply_commands(base, command, i, overlay)
            options[i] = {"img": out, "cmds": [f"photo {i % N + 1}: {cmd or '(default variation)'}"]}
            if i == 0:
                notes = nt
        msg = f"Created {n} option(s) from {N} photo(s)."
        if notes:
            msg += " | Notes: " + "; ".join(notes)
    state["options"] = options
    msg += " Unsaved options from before were replaced."
    return ([state] + option_images(state)
            + [msg, gr.update(choices=overlay_choices(state), value="None"), gr.update()])


# ---------- STEP 2: edit one option ----------
def open_edit(slot, state):
    opt = state["options"][slot]
    if opt is None:
        return [state, gr.update(), gr.update(), None, "", f"Option {slot + 1} is empty."]
    state.update(slot=slot, orig={"img": opt["img"].copy(), "cmds": list(opt["cmds"])},
                 work=opt["img"].copy(), cmds=list(opt["cmds"]), undo=[])
    return [state, gr.update(visible=False), gr.update(visible=True), state["work"],
            edit_log(state), f"Editing option {slot + 1}."]


def run_tool(state, label, fn):
    """Apply fn(image) to the working image, with undo + history."""
    if state["work"] is None:
        return [state, None, "", "Open an option to edit first."]
    state["undo"].append((state["work"].copy(), list(state["cmds"])))
    try:
        out = fn(state["work"]).convert("RGB")
    except Exception as e:
        state["undo"].pop()
        return [state, state["work"], edit_log(state), f"Could not apply {label}: {e}"]
    state["work"] = out
    state["cmds"].append(label)
    return [state, out, edit_log(state), f"Applied: {label}"]


def apply_prompt(command, overlay_choice, mode, state):
    if state["work"] is None:
        return [state, None, "", "Open an option to edit first."]
    command = (command or "").strip()
    if not command:
        return [state, state["work"], edit_log(state), "Type an edit instruction first."]
    if mode == AI_MODE:
        return run_tool(state, f"AI: {command}", lambda im: ai_edit(im, command))
    overlay = get_overlay(state, overlay_choice)
    holder = {}

    def fn(im):
        out, notes = apply_commands(im, command, 0, overlay)
        holder["notes"] = notes
        return out

    before = state["work"].copy()
    result = run_tool(state, command, fn)
    notes = holder.get("notes")
    if notes:
        try:
            unchanged = (before.size == state["work"].size and
                         ImageChops.difference(before, state["work"]).getbbox() is None)
        except Exception:
            unchanged = False
        if unchanged and state["undo"]:  # nothing recognized: don't log a fake edit
            state["work"], state["cmds"] = state["undo"].pop()
            result[1], result[2] = state["work"], edit_log(state)
            result[3] = "No change: " + "; ".join(notes) + ". Try the tool tabs below."
        else:
            result[3] += " | " + "; ".join(notes)
    return result


def undo_edit(state):
    if not state["undo"]:
        return [state, state["work"], edit_log(state), "Nothing to undo."]
    state["work"], state["cmds"] = state["undo"].pop()
    return [state, state["work"], edit_log(state), "Undid last edit."]


def reset_edit(state):
    if state["orig"] is None:
        return [state, None, "", "Nothing to reset."]
    state.update(work=state["orig"]["img"].copy(), cmds=list(state["orig"]["cmds"]), undo=[])
    return [state, state["work"], edit_log(state), "Reset to how it was when you opened it."]


def leave_edit(state, save):
    slot = state["slot"]
    if save and slot is not None and state["work"] is not None:
        state["options"][slot] = {"img": state["work"].copy(), "cmds": list(state["cmds"])}
        msg = f"Saved changes to option {slot + 1}."
    else:
        msg = "Left editor without saving."
    state.update(slot=None, orig=None, work=None, cmds=[], undo=[])
    return ([state, gr.update(visible=True), gr.update(visible=False)]
            + option_images(state) + [msg])


# ----- one-click tools -----
QUICK_TOOLS = [
    ("Rotate left", lambda im: im.rotate(90, expand=True)),
    ("Rotate right", lambda im: im.rotate(-90, expand=True)),
    ("Flip horizontal", lambda im: ImageOps.mirror(im)),
    ("Flip vertical", lambda im: ImageOps.flip(im)),
    ("Grayscale", lambda im: ImageOps.grayscale(im)),
    ("Invert", lambda im: ImageOps.invert(im)),
    ("Auto contrast", lambda im: ImageOps.autocontrast(im, cutoff=2)),
    ("Equalize", lambda im: ImageOps.equalize(im)),
    ("Sharpen", lambda im: im.filter(ImageFilter.UnsharpMask(radius=2, percent=150))),
    ("Blur", lambda im: im.filter(ImageFilter.GaussianBlur(3))),
    ("Smooth", lambda im: im.filter(ImageFilter.SMOOTH_MORE)),
    ("Edges", lambda im: im.filter(ImageFilter.FIND_EDGES)),
    ("Emboss", lambda im: im.filter(ImageFilter.EMBOSS)),
    ("Contour", lambda im: im.filter(ImageFilter.CONTOUR)),
]


# ----- crop & size -----
def crop_box(im, l, t, r, b):
    w, h = im.size
    box = (int(w * l / 100), int(h * t / 100), int(w * (1 - r / 100)), int(h * (1 - b / 100)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise ValueError("crop area too small")
    return im.crop(box)


def crop_aspect(im, aw, ah):
    w, h = im.size
    target = aw / ah
    if w / h > target:
        nw = int(h * target)
        left = (w - nw) // 2
        return im.crop((left, 0, left + nw, h))
    nh = int(w / target)
    top = (h - nh) // 2
    return im.crop((0, top, w, top + nh))


def t_crop(l, t, r, b, state):
    return run_tool(state, f"crop (left {l}%, top {t}%, right {r}%, bottom {b}%)",
                    lambda im: crop_box(im, l, t, r, b))


def t_resize(pct, state):
    return run_tool(state, f"resize {int(pct)}%",
                    lambda im: im.resize((max(1, int(im.width * pct / 100)),
                                          max(1, int(im.height * pct / 100))), Image.LANCZOS))


def t_rotate(angle, state):
    return run_tool(state, f"rotate {int(angle)}°",
                    lambda im: im.rotate(angle, expand=True, resample=Image.BICUBIC))


def t_border(px, state):
    return run_tool(state, f"border {int(px)}px",
                    lambda im: ImageOps.expand(im, border=int(px), fill="white"))


# ----- adjust -----
def t_adjust(bright, contrast, color, sharp, state):
    parts = []

    def fn(im):
        if bright != 1:
            im = ImageEnhance.Brightness(im).enhance(bright); parts.append(f"brightness {bright}")
        if contrast != 1:
            im = ImageEnhance.Contrast(im).enhance(contrast); parts.append(f"contrast {contrast}")
        if color != 1:
            im = ImageEnhance.Color(im).enhance(color); parts.append(f"saturation {color}")
        if sharp != 1:
            im = ImageEnhance.Sharpness(im).enhance(sharp); parts.append(f"sharpness {sharp}")
        if not parts:
            raise ValueError("move at least one slider away from 1.0")
        return im

    result = run_tool(state, "adjust", fn)
    if state["cmds"] and state["cmds"][-1] == "adjust":
        state["cmds"][-1] = "adjust: " + ", ".join(parts)
        result[2] = edit_log(state)
        result[3] = "Applied: " + state["cmds"][-1]
    return result


# ----- text -----
def t_text(txt, size, pos, state):
    def fn(im):
        if not (txt or "").strip():
            raise ValueError("type some text first")
        im = im.copy()
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=int(size))
        box = d.textbbox((0, 0), txt, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = max(0, (im.width - tw) // 2)
        margin = int(im.height * 0.05)
        y = {"Top": margin, "Center": (im.height - th) // 2,
             "Bottom": im.height - th - margin}[pos]
        d.text((x, y), txt, fill="white", font=font,
               stroke_width=max(1, int(size) // 15), stroke_fill="black")
        return im
    return run_tool(state, f"text '{txt}'", fn)


# ----- layers -----
def t_blend(choice, alpha, state):
    other = get_overlay(state, choice)

    def fn(im):
        if other is None:
            raise ValueError("pick an overlay photo first")
        return Image.blend(im, other.convert("RGB").resize(im.size), alpha)
    return run_tool(state, f"blend with {choice} at {alpha}", fn)


def t_paste(choice, corner, pct, state):
    other = get_overlay(state, choice)

    def fn(im):
        if other is None:
            raise ValueError("pick an overlay photo first")
        im = im.copy()
        small = other.copy()
        small.thumbnail((max(1, int(im.width * pct / 100)), max(1, int(im.height * pct / 100))))
        m = 20
        x = m if "left" in corner else im.width - small.width - m
        y = m if corner.startswith("Top") else im.height - small.height - m
        im.paste(small, (x, y))
        return im
    return run_tool(state, f"paste {choice} ({corner})", fn)


# ---------- finalize actions ----------
def save_slot(slot, state):
    opt = state["options"][slot]
    if opt is None:
        gallery, text = final_render(state)
        return [state] + option_images(state) + [gallery, text, f"Option {slot + 1} is empty."]
    state["counter"] += 1
    state["final"].append({"id": state["counter"], "img": opt["img"].copy(), "cmds": list(opt["cmds"])})
    state["options"][slot] = None
    gallery, text = final_render(state)
    return ([state] + option_images(state)
            + [gallery, text, f"Option {slot + 1} saved as #{state['counter']} and moved to Finalized."])


def discard_slot(slot, state):
    state["options"][slot] = None
    return [state] + option_images(state) + [f"Discarded option {slot + 1}."]


def toggle_view(mode):
    return gr.update(visible=mode == "Thumbnails"), gr.update(visible=mode == "Text")


OPEN_TAB_JS = "(img) => { if (img && img.url) { window.open(img.url, '_blank'); } }"

HELP = """**AI edit mode:** describe the change in plain English, e.g. `remove the toys from the floor` or `remove the outer onion layer`. Results can vary, so generate again or keep editing in the editor.\n\n**Quick commands mode** (keyword-based, not AI). Combine with commas or "then", e.g. `grayscale, blur 3, text Hello`

- **Adjust:** grayscale, brighten, darker, contrast, saturation, autocontrast, equalize, posterize, invert, solarize
- **Filter:** blur, sharpen, edges, emboss, contour, smooth, detail
- **Draw:** text <words>, box, watermark
- **Geometry:** rotate left / right / 45, mirror, flip horizontal / vertical, crop 70, resize 50%, thumbnail 256, fit 512, pad 512, expand 30
- **Layers (uses your other photos):** blend 0.5, composite 120, paste
- **Chops (vs. original):** chops difference, chops multiply, chops lighter, chops darker

In the editor, the tool tabs give you buttons and sliders for the same things, plus real crop controls.
"""

# ---------- UI ----------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State(new_state())
    gr.Markdown("# ⛽ FuelStrictStudio")
    status_output = gr.Textbox(show_label=False, interactive=False, lines=1,
                               placeholder="Status messages appear here")

    # ===== STEP 1 view =====
    with gr.Column(visible=True) as create_view:
        command_input = gr.Textbox(label="Prompt",
                                   placeholder="AI: 'remove the toys from the floor'  |  Quick: 'grayscale, blur 3'")
        mode_create = gr.Radio(MODES, value=DEFAULT_MODE, label="Edit mode",
                               info="AI edit understands plain English (needs GEMINI_API_KEY). Quick commands are keyword tools.")
        photo_files = gr.File(label="Upload one or more photos (up to 10)",
                              file_count="multiple", file_types=["image"])
        with gr.Row():
            num_outputs = gr.Slider(1, 3, step=1, value=3, label="Options to create")
            run_btn = gr.Button("Generate options", variant="primary")
        gr.Markdown("_With several photos, each option uses a different photo; the others become layers for blend / composite / paste._")
        with gr.Accordion("Prompt tips", open=False):
            gr.Markdown(HELP)

        gr.Markdown("### Options")
        slot_imgs, slot_btns = [], []
        for n in range(3):
            with gr.Group():
                gr.Markdown(f"**Option {n + 1}**")
                img = gr.Image(type="pil", label=f"Option {n + 1}", interactive=False)
                with gr.Row():
                    b_edit = gr.Button("Edit", size="sm")
                    b_save = gr.Button("Save", size="sm", variant="primary")
                    b_tab = gr.Button("New tab", size="sm")
                    b_discard = gr.Button("Discard", size="sm", variant="stop")
            slot_imgs.append(img)
            slot_btns.append((b_edit, b_save, b_tab, b_discard))

    # ===== STEP 2 view =====
    with gr.Column(visible=False) as edit_view:
        gr.Markdown("## Editor")
        edit_cmd = gr.Textbox(label="Edit prompt (each edit builds on the last)",
                              placeholder="AI: 'remove the outer onion layer'  |  Quick: 'rotate right, darker'")
        mode_edit = gr.Radio(MODES, value=DEFAULT_MODE, label="Edit mode")
        with gr.Row():
            apply_btn = gr.Button("Apply", variant="primary")
            undo_btn = gr.Button("Undo")
            reset_btn = gr.Button("Reset")
        edit_img = gr.Image(type="pil", label="Image being edited", interactive=False, height=360)

        with gr.Tabs():
            with gr.Tab("Quick"):
                quick_btns = []
                for k in range(0, len(QUICK_TOOLS), 3):
                    with gr.Row():
                        for label, fn in QUICK_TOOLS[k:k + 3]:
                            quick_btns.append((gr.Button(label, size="sm"), label, fn))

            with gr.Tab("Crop & size"):
                gr.Markdown("**Crop** — percent to cut from each side")
                crop_l = gr.Slider(0, 45, value=0, step=1, label="Left %")
                crop_t = gr.Slider(0, 45, value=0, step=1, label="Top %")
                crop_r = gr.Slider(0, 45, value=0, step=1, label="Right %")
                crop_b = gr.Slider(0, 45, value=0, step=1, label="Bottom %")
                crop_btn = gr.Button("Crop", variant="primary")
                gr.Markdown("**Crop to shape** (centered)")
                with gr.Row():
                    sq_btn = gr.Button("Square 1:1", size="sm")
                    r43_btn = gr.Button("4:3", size="sm")
                    r169_btn = gr.Button("16:9", size="sm")
                    p916_btn = gr.Button("9:16", size="sm")
                gr.Markdown("**Resize / rotate / border**")
                resize_pct = gr.Slider(10, 200, value=100, step=5, label="Resize %")
                resize_btn = gr.Button("Resize")
                rot_angle = gr.Slider(-180, 180, value=0, step=1, label="Rotate degrees")
                rot_btn = gr.Button("Rotate")
                border_px = gr.Slider(1, 100, value=20, step=1, label="Border px")
                border_btn = gr.Button("Add border")

            with gr.Tab("Adjust"):
                adj_b = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label="Brightness")
                adj_c = gr.Slider(0.2, 2.5, value=1.0, step=0.05, label="Contrast")
                adj_s = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Saturation")
                adj_sh = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Sharpness")
                adj_btn = gr.Button("Apply adjustments", variant="primary")

            with gr.Tab("Text"):
                txt_in = gr.Textbox(label="Text")
                txt_size = gr.Slider(16, 200, value=56, step=2, label="Size")
                txt_pos = gr.Radio(["Top", "Center", "Bottom"], value="Bottom", label="Position")
                txt_btn = gr.Button("Add text", variant="primary")

            with gr.Tab("Layers"):
                overlay_choice = gr.Dropdown(["None"], value="None", label="Overlay photo (from your uploads)")
                blend_alpha = gr.Slider(0.05, 0.95, value=0.4, step=0.05, label="Blend amount")
                blend_btn = gr.Button("Blend")
                paste_corner = gr.Dropdown(["Top-left", "Top-right", "Bottom-left", "Bottom-right"],
                                           value="Top-right", label="Paste corner")
                paste_pct = gr.Slider(10, 60, value=30, step=5, label="Pasted size %")
                paste_btn = gr.Button("Paste small copy")

        gr.Markdown("**Edit history**")
        edit_history = gr.Markdown("_No edits yet._")
        with gr.Row():
            save_back_btn = gr.Button("Save changes & back", variant="primary")
            cancel_btn = gr.Button("Back without saving")

    # ===== STEP 3: finalized space =====
    gr.Markdown("### Finalized images (this session)")
    view_mode = gr.Radio(["Thumbnails", "Text"], value="Thumbnails", label="View")
    final_gallery = gr.Gallery(label="Finalized", columns=3, object_fit="contain",
                               height="auto", allow_preview=True, visible=True)
    final_text = gr.Markdown("_Nothing finalized yet._", visible=False)

    # ----- wiring: step 1 -----
    run_btn.click(generate, [photo_files, command_input, mode_create, num_outputs, state],
                  [state] + slot_imgs + [status_output, overlay_choice, final_gallery])

    for n, (img, (b_edit, b_save, b_tab, b_discard)) in enumerate(zip(slot_imgs, slot_btns)):
        b_edit.click(lambda st, n=n: open_edit(n, st), [state],
                     [state, create_view, edit_view, edit_img, edit_history, status_output])
        b_save.click(lambda st, n=n: save_slot(n, st), [state],
                     [state] + slot_imgs + [final_gallery, final_text, status_output])
        b_tab.click(fn=None, inputs=[img], outputs=None, js=OPEN_TAB_JS)
        b_discard.click(lambda st, n=n: discard_slot(n, st), [state],
                        [state] + slot_imgs + [status_output])

    # ----- wiring: step 2 -----
    E = [state, edit_img, edit_history, status_output]
    apply_btn.click(apply_prompt, [edit_cmd, overlay_choice, mode_edit, state], E)
    edit_cmd.submit(apply_prompt, [edit_cmd, overlay_choice, mode_edit, state], E)
    undo_btn.click(undo_edit, [state], E)
    reset_btn.click(reset_edit, [state], E)

    for btn, label, fn in quick_btns:
        btn.click(lambda st, label=label, fn=fn: run_tool(st, label.lower(), fn), [state], E)

    crop_btn.click(t_crop, [crop_l, crop_t, crop_r, crop_b, state], E)
    for btn, (aw, ah) in ((sq_btn, (1, 1)), (r43_btn, (4, 3)), (r169_btn, (16, 9)), (p916_btn, (9, 16))):
        btn.click(lambda st, aw=aw, ah=ah: run_tool(st, f"crop to {aw}:{ah}",
                                                   lambda im: crop_aspect(im, aw, ah)), [state], E)
    resize_btn.click(t_resize, [resize_pct, state], E)
    rot_btn.click(t_rotate, [rot_angle, state], E)
    border_btn.click(t_border, [border_px, state], E)
    adj_btn.click(t_adjust, [adj_b, adj_c, adj_s, adj_sh, state], E)
    txt_btn.click(t_text, [txt_in, txt_size, txt_pos, state], E)
    blend_btn.click(t_blend, [overlay_choice, blend_alpha, state], E)
    paste_btn.click(t_paste, [overlay_choice, paste_corner, paste_pct, state], E)

    leave_outputs = [state, create_view, edit_view] + slot_imgs + [status_output]
    save_back_btn.click(lambda st: leave_edit(st, True), [state], leave_outputs)
    cancel_btn.click(lambda st: leave_edit(st, False), [state], leave_outputs)

    # ----- wiring: step 3 -----
    view_mode.change(toggle_view, [view_mode], [final_gallery, final_text])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
