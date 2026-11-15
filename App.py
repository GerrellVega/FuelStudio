import os
import re
import random
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
    return img.rotate(num(p, 90 * (i + 1)), expand=True, resample=Image.BICUBIC)


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
    (["autocontrast"], op_autocontrast),
    (["grayscale", "black and white"], op_grayscale),
    (["bright"], op_brighten),
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
    (["difference", "diff"], op_difference),
    (["multiply"], op_multiply),
    (["lighter"], op_lighter),
    (["darker"], op_darker),
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


# ---------- session state ----------
def new_state():
    return {"options": [None, None, None], "final": [], "slot": None,
            "orig": None, "work": None, "cmds": [], "undo": [], "counter": 0}


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


# ---------- STEP 1: create 1-3 options ----------
def generate(image, overlay, command, n, state):
    if image is None:
        return [state] + option_images(state) + ["Please upload a source image first."]
    base = image.convert("RGB")
    options, notes = [None, None, None], []
    for i in range(int(n)):
        out, nt = apply_commands(base, command, i, overlay)
        options[i] = {"img": out, "cmds": [(command or "").strip() or "(default variation)"]}
        if i == 0:
            notes = nt
    state["options"] = options
    msg = f"Created {int(n)} option(s) for '{command}'. Previous unsaved options were replaced."
    if notes:
        msg += " | Notes: " + "; ".join(notes)
    return [state] + option_images(state) + [msg]


# ---------- STEP 2: edit one option with sequenced edits ----------
def open_edit(slot, state):
    opt = state["options"][slot]
    if opt is None:
        return [state, gr.update(), gr.update(), None, "", f"Slot {slot + 1} is empty."]
    state.update(slot=slot, orig={"img": opt["img"].copy(), "cmds": list(opt["cmds"])},
                 work=opt["img"].copy(), cmds=list(opt["cmds"]), undo=[])
    return [state, gr.update(visible=False), gr.update(visible=True), state["work"],
            edit_log(state), f"Editing option {slot + 1}."]


def apply_edit(command, overlay, state):
    if state["work"] is None:
        return [state, None, "", "Open an option to edit first."]
    if not (command or "").strip():
        return [state, state["work"], edit_log(state), "Type an edit instruction first."]
    state["undo"].append((state["work"].copy(), list(state["cmds"])))
    out, notes = apply_commands(state["work"], command, 0, overlay)
    state["work"] = out
    state["cmds"].append(command.strip())
    msg = f"Applied '{command.strip()}' (edit {len(state['cmds'])})."
    if notes:
        msg += " | Notes: " + "; ".join(notes)
    return [state, out, edit_log(state), msg]


def undo_edit(state):
    if not state["undo"]:
        return [state, state["work"], edit_log(state), "Nothing to undo."]
    state["work"], state["cmds"] = state["undo"].pop()
    return [state, state["work"], edit_log(state), "Undid last edit."]


def reset_edit(state):
    if state["orig"] is None:
        return [state, None, "", "Nothing to reset."]
    state.update(work=state["orig"]["img"].copy(), cmds=list(state["orig"]["cmds"]), undo=[])
    return [state, state["work"], edit_log(state), "Reset to the option as it was when you opened it."]


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


# ---------- finalize actions per option ----------
def save_slot(slot, state):
    opt = state["options"][slot]
    if opt is None:
        gallery, text = final_render(state)
        return [state] + option_images(state) + [gallery, text, f"Slot {slot + 1} is empty."]
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

HELP = """**Commands** (combine with commas or "then", e.g. `grayscale, blur 3, text Hello`)

- **Adjust:** grayscale, brighten, contrast, autocontrast, equalize, posterize, invert, solarize
- **Filter:** blur, sharpen, edges, emboss, contour, smooth, detail
- **Draw:** text <words>, box, watermark
- **Chops (vs. original):** difference, multiply, lighter, darker
- **Geometry:** rotate 45, transpose, mirror, flip, crop 70, resize 50% (or width), thumbnail 256, fit 512, pad 512, expand 30
- **Layers (use the overlay image):** blend 0.5, composite 120, paste
- **Grab:** grab (server screen capture; won't work on Render)
"""

# ---------- UI ----------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State(new_state())
    gr.Markdown("# ⛽ FuelStrictStudio")

    # ===== STEP 1 view =====
    with gr.Column(visible=True) as create_view:
        command_input = gr.Textbox(label="Prompt", placeholder="e.g., 'grayscale, blur 3, text Hello'")
        with gr.Row():
            input_image = gr.Image(type="pil", label="Source image")
            overlay_image = gr.Image(type="pil", label="Optional overlay image")
        with gr.Row():
            num_outputs = gr.Slider(1, 3, step=1, value=3, label="Options to create")
            run_btn = gr.Button("Generate options", variant="primary")
        with gr.Accordion("Command reference", open=False):
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
        edit_cmd = gr.Textbox(label="Edit prompt (each edit builds on the last)",
                              placeholder="e.g., 'rotate 90' then 'blur 2' then 'text Done'")
        edit_overlay = gr.Image(type="pil", label="Optional overlay image")
        with gr.Row():
            apply_btn = gr.Button("Apply edit", variant="primary")
            undo_btn = gr.Button("Undo")
            reset_btn = gr.Button("Reset")
        edit_img = gr.Image(type="pil", label="Image being edited", interactive=False)
        gr.Markdown("**Edit history**")
        edit_history = gr.Markdown("_No edits yet._")
        with gr.Row():
            save_back_btn = gr.Button("Save changes & back", variant="primary")
            cancel_btn = gr.Button("Back without saving")

    status_output = gr.Textbox(label="Status", interactive=False)

    # ===== STEP 3: finalized space =====
    gr.Markdown("### Finalized images (this session)")
    view_mode = gr.Radio(["Thumbnails", "Text"], value="Thumbnails", label="View")
    final_gallery = gr.Gallery(label="Finalized", columns=3, object_fit="contain",
                               height="auto", allow_preview=True, visible=True)
    final_text = gr.Markdown("_Nothing finalized yet._", visible=False)

    # ----- wiring -----
    run_btn.click(generate, [input_image, overlay_image, command_input, num_outputs, state],
                  [state] + slot_imgs + [status_output])

    for n, (img, (b_edit, b_save, b_tab, b_discard)) in enumerate(zip(slot_imgs, slot_btns)):
        b_edit.click(lambda st, n=n: open_edit(n, st), [state],
                     [state, create_view, edit_view, edit_img, edit_history, status_output])
        b_save.click(lambda st, n=n: save_slot(n, st), [state],
                     [state] + slot_imgs + [final_gallery, final_text, status_output])
        b_tab.click(fn=None, inputs=[img], outputs=None, js=OPEN_TAB_JS)
        b_discard.click(lambda st, n=n: discard_slot(n, st), [state],
                        [state] + slot_imgs + [status_output])

    edit_outputs = [state, edit_img, edit_history, status_output]
    apply_btn.click(apply_edit, [edit_cmd, edit_overlay, state], edit_outputs)
    edit_cmd.submit(apply_edit, [edit_cmd, edit_overlay, state], edit_outputs)
    undo_btn.click(undo_edit, [state], edit_outputs)
    reset_btn.click(reset_edit, [state], edit_outputs)

    leave_outputs = [state, create_view, edit_view] + slot_imgs + [status_output]
    save_back_btn.click(lambda st: leave_edit(st, True), [state], leave_outputs)
    cancel_btn.click(lambda st: leave_edit(st, False), [state], leave_outputs)

    view_mode.change(toggle_view, [view_mode], [final_gallery, final_text])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
