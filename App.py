import io
import os
import re
import random
import urllib.request

import numpy as np
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


# ---------- limits (tuned to keep memory low) ----------
MAX_PHOTOS = 12          # most photos accepted per upload
WORK_SIDE = 1536         # uploads are shrunk to this long side
REF_SIDE = 1024          # reference photos sent to the AI are shrunk to this
UNDO_LIMIT = 10          # undo steps kept per edit session
MAX_REFS = max(1, int(os.environ.get("GEMINI_MAX_REFS", "3")))  # images per AI request, incl. the base


def shrink(img, side):
    out = img.convert("RGB")
    out.thumbnail((side, side))
    return out


def mentioned_photos(text, n_photos):
    """Photo numbers named in a prompt, e.g. 'put the sofa from photo 2 into photo 1' -> [2, 1]."""
    nums = []
    for m in re.findall(r"photo\s*#?\s*(\d+)", text or "", flags=re.I):
        k = int(m)
        if 1 <= k <= n_photos and k not in nums:
            nums.append(k)
    return nums


# ---------- AI editing (Google Gemini image model) ----------
AI_MODE = "AI edit"
QUICK_MODE = "Quick commands"
COMPOSE_MODE = "Compose (local, no AI)"
MODES = [AI_MODE, QUICK_MODE, COMPOSE_MODE]
EDIT_MODES = [AI_MODE, QUICK_MODE]
DEFAULT_MODE = AI_MODE
GEMINI_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
_client = None


def using_gemini():
    return bool(os.environ.get("GEMINI_API_KEY"))


def max_images():
    """Images per AI request: several with Gemini, one with the free Hugging Face Space."""
    return MAX_REFS if using_gemini() else 1


# ---- free option: a public Hugging Face Space (FLUX.1 Kontext), called through its queue ----
HF_SPACE = os.environ.get("HF_SPACE", "black-forest-labs/FLUX.1-Kontext-Dev")
HF_API_NAME = os.environ.get("HF_API_NAME", "/infer")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "300"))
_hf_client = None


def hf_edit(img, instruction):
    """Edit one image with plain-English instructions via a free Hugging Face Space."""
    global _hf_client
    import tempfile
    from gradio_client import Client, handle_file
    if not (instruction or "").strip():
        raise RuntimeError("type what you want changed")
    if _hf_client is None:
        token = os.environ.get("HF_TOKEN")
        attempts = [{"token": token}, {"hf_token": token}] if token else [{}]
        last = None
        for kw in attempts:
            try:
                _hf_client = Client(HF_SPACE, **kw)
                break
            except TypeError as e:  # different gradio_client versions name the token differently
                last = e
        if _hf_client is None:
            raise RuntimeError(f"could not connect to {HF_SPACE}: {last}")
    src = shrink(img, REF_SIDE)  # smaller input = less free GPU time used
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        src.save(tmp, format="PNG")
        path = tmp.name
    try:
        job = _hf_client.submit(input_image=handle_file(path), prompt=instruction.strip(),
                                seed=0, randomize_seed=True, guidance_scale=2.5, steps=28,
                                api_name=HF_API_NAME)
        result = job.result(timeout=HF_TIMEOUT)  # waits in the Space's queue
    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower():
            msg += " (free GPU quota used up: wait, or set a free HF_TOKEN)"
        raise RuntimeError(msg[:220])
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    first = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(first, dict):
        first = first.get("path") or first.get("value")
    if not first:
        raise RuntimeError("the Space returned no image")
    return Image.open(first).convert("RGB")


def ai_edit(base, instruction, base_label="the photo to edit", refs=None):
    """Gemini if GEMINI_API_KEY is set (supports reference photos), else the free HF Space."""
    if using_gemini():
        return gemini_edit(base, instruction, base_label, refs)
    return hf_edit(base, instruction)


def gemini_edit(base, instruction, base_label="the photo to edit", refs=None):
    """Send the base image (+ optional labeled reference images) and an instruction to Gemini."""
    global _client
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set on the server")
    if not (instruction or "").strip():
        raise RuntimeError("type what you want changed")
    from google import genai
    if _client is None:
        _client = genai.Client(api_key=key)
    refs = refs or []
    images = [shrink(base, WORK_SIDE)] + [shrink(r, REF_SIDE) for _, r in refs]
    desc = [f"Image 1 is {base_label}."]
    desc += [f"Image {j + 2} is {label}." for j, (label, _) in enumerate(refs)]
    prompt = (" ".join(desc) + " Produce one edited version of Image 1 following the instruction. "
              "Keep everything else the same unless the instruction needs it changed: same scene, "
              "framing, lighting and style. Instruction: " + instruction.strip())
    resp = _client.models.generate_content(model=GEMINI_MODEL, contents=[prompt] + images)
    texts = []
    for part in (getattr(resp, "parts", None) or []):
        data = getattr(getattr(part, "inline_data", None), "data", None)
        if data:
            return Image.open(io.BytesIO(data)).convert("RGB")
        if getattr(part, "text", None):
            texts.append(part.text)
    detail = " ".join(texts)[:200] if texts else "The model may have declined this request."
    raise RuntimeError("No image came back. " + detail)


# ---------- local scan: YOLO via onnxruntime ----------
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
_yolo = None


def get_yolo():
    global _yolo
    if _yolo is None:
        path = os.environ.get("YOLO_PATH", "models/yolov8n.onnx")
        if not os.path.exists(path):
            raise RuntimeError(f"YOLO model file not found at '{path}'. Run export_yolo.py once "
                               "and commit models/yolov8n.onnx to GitHub.")
        import onnxruntime as ort
        _yolo = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _yolo


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a_i + a_r - inter + 1e-9)
        order = rest[iou < iou_thr]
    return keep


def detect_objects(img, conf_thr=0.25, iou_thr=0.45, size=640):
    """Returns [{'label', 'conf', 'box': (x0, y0, x1, y1)}] in the image's own pixels."""
    sess = get_yolo()
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (px, py))
    x = (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    pred = out[0]
    if pred.shape[0] < pred.shape[1]:  # (84, 8400) -> (8400, 84)
        pred = pred.T
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf > conf_thr
    pred, cls, conf = pred[keep], cls[keep], conf[keep]
    if len(pred) == 0:
        return []
    b = pred[:, :4]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - px) / scale).clip(0, w)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - py) / scale).clip(0, h)
    offset = cls[:, None] * 10000.0  # class-aware NMS
    idx = nms(xyxy + offset, conf, iou_thr)[:30]
    return [{"label": COCO[int(cls[i])] if int(cls[i]) < len(COCO) else f"class {int(cls[i])}",
             "conf": float(conf[i]), "box": tuple(float(v) for v in xyxy[i])} for i in idx]


def draw_detections(img, dets):
    out = img.copy()
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=max(14, img.width // 40))
    for k, det in enumerate(dets):
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0, y0, x1, y1], outline="red", width=max(2, img.width // 300))
        d.text((x0 + 3, y0 + 2), str(k + 1), fill="yellow", font=font,
               stroke_width=2, stroke_fill="black")
    return out


# ---------- local removal: LaMa via onnxruntime ----------
LAMA_URL = os.environ.get("LAMA_URL",
                          "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx")
_lama = None


def get_lama():
    global _lama
    if _lama is None:
        path = os.environ.get("LAMA_PATH", "models/lama_fp32.onnx")
        if not os.path.exists(path):  # ~208 MB, downloaded once per deploy
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = path + ".part"
            urllib.request.urlretrieve(LAMA_URL, tmp)
            os.replace(tmp, path)
        import onnxruntime as ort
        _lama = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _lama


def lama_inpaint(img, mask):
    """Erase the white area of `mask` (mode 'L', same size as img) and fill from surroundings."""
    img = img.convert("RGB")
    m = np.asarray(mask.convert("L")) > 127
    ys, xs = np.where(m)
    if len(xs) == 0:
        raise ValueError("nothing selected to remove")
    sess = get_lama()
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    side = max(256, int(max(x1 - x0, y1 - y0) * 2))   # crop with plenty of context
    cw, ch = min(side, img.width), min(side, img.height)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = min(max(cx - cw // 2, 0), img.width - cw)
    top = min(max(cy - ch // 2, 0), img.height - ch)
    box = (left, top, left + cw, top + ch)
    crop, mcrop = img.crop(box), mask.convert("L").crop(box)

    a = (np.asarray(crop.resize((512, 512), Image.BICUBIC), np.float32) / 255.0).transpose(2, 0, 1)[None]
    mk = (np.asarray(mcrop.resize((512, 512), Image.NEAREST), np.float32) > 127).astype(np.float32)[None, None]
    names = [i.name for i in sess.get_inputs()]
    out = sess.run(None, {names[0]: a, names[1]: mk})[0][0]
    filled = Image.fromarray(np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8))
    filled = filled.resize(crop.size, Image.BICUBIC)

    soft = mcrop.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
    merged = Image.composite(filled, crop, soft)   # only the masked area changes
    result = img.copy()
    result.paste(merged, (left, top))
    return result


# ---------- optional 2x upscale on save: Real-ESRGAN via onnxruntime ----------
UPSCALE_PATH = os.environ.get("ESRGAN_PATH", "models/realesr_x2.onnx")
UPSCALE_TILE = int(os.environ.get("ESRGAN_TILE", "256"))   # small tiles keep memory low
UPSCALE_PAD = 16                                            # overlap context for clean tile edges
UPSCALE_MAX_SIDE = int(os.environ.get("ESRGAN_MAX_SIDE", "2048"))  # refuse bigger inputs
_esrgan = None


def get_esrgan():
    global _esrgan
    if _esrgan is None:
        if not os.path.exists(UPSCALE_PATH):
            raise RuntimeError(f"upscaler file not found at '{UPSCALE_PATH}'. Run export_esrgan.py "
                               "once and commit models/realesr_x2.onnx to GitHub.")
        import onnxruntime as ort
        _esrgan = ort.InferenceSession(UPSCALE_PATH, providers=["CPUExecutionProvider"])
    return _esrgan


def upscale2x(img):
    """Double width and height with Real-ESRGAN, one tile at a time to limit memory."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > UPSCALE_MAX_SIDE:
        raise ValueError(f"image is too large to upscale (longest side over {UPSCALE_MAX_SIDE}px)")
    sess = get_esrgan()
    in_name = sess.get_inputs()[0].name
    src = np.asarray(img, np.uint8)
    out = np.zeros((h * 2, w * 2, 3), np.uint8)
    T, P = UPSCALE_TILE, UPSCALE_PAD
    for y in range(0, h, T):
        for x in range(0, w, T):
            x1, y1 = min(x + T, w), min(y + T, h)                 # core of this tile
            px0, py0 = max(x - P, 0), max(y - P, 0)               # padded input region
            px1, py1 = min(x1 + P, w), min(y1 + P, h)
            tile = src[py0:py1, px0:px1].astype(np.float32) / 255.0
            inp = np.ascontiguousarray(tile.transpose(2, 0, 1)[None])
            res = sess.run(None, {in_name: inp})[0][0].transpose(1, 2, 0)
            cx0, cy0 = (x - px0) * 2, (y - py0) * 2                # crop padding back off
            cx1, cy1 = cx0 + (x1 - x) * 2, cy0 + (y1 - y) * 2
            core = res[cy0:cy1, cx0:cx1] * 255.0 + 0.5
            out[y * 2:y1 * 2, x * 2:x1 * 2] = np.clip(core, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ---------- session state ----------
def new_state():
    return {"options": [None, None, None], "final": [], "photos": [], "detections": [],
            "slot": None, "orig": None, "work": None, "cmds": [], "undo": [], "counter": 0,
            "people": {}, "cutouts": {}, "chosen": {}, "placements": {},
            "size_frac": 0.45, "setup_names": None}


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
    """Open uploads as upright RGB images, shrunk to WORK_SIDE, capped at MAX_PHOTOS."""
    if not files:
        return [], 0
    if not isinstance(files, (list, tuple)):
        files = [files]
    photos = []
    for f in files[:MAX_PHOTOS]:
        path = getattr(f, "name", f)
        try:
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # fix phone rotation
            im.thumbnail((WORK_SIDE, WORK_SIDE))
            photos.append(im)
        except Exception:
            continue
    return photos, max(0, len(files) - MAX_PHOTOS)


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
    photos, skipped = load_photos(files)
    if not photos:
        return ([state] + option_images(state) + ["Upload at least one photo first.",
                gr.update(), gr.update()])
    n = int(n)
    N = len(photos)
    state["photos"] = photos
    options, notes, errors = [None, None, None], [], []
    cmd = (command or "").strip()
    extra = f" Only the first {MAX_PHOTOS} photos were used." if skipped else ""

    if mode == COMPOSE_MODE:
        ok, why = compose_ready(files, state)
        if not ok:
            return [state] + option_images(state) + [why, gr.update(), gr.update()]
        srcs = compose_sources(state)
        pend = pending_photos(state)
        options = [None, None, None]
        for i in range(n):
            out = compose_cycle(photos[0], srcs, state, i)
            label = ", ".join(f"photo {k}" for k, _ in srcs)
            options[i] = {"img": out, "cmds": [f"compose {i + 1}: base photo 1 + {label}"]}
        notes = []
        msg = f"Composed {n} independent option(s) from {len(srcs)} person(s)."
        if pend:
            msg += " Skipped unchecked photo(s): " + ", ".join(str(k) for k in pend) + "."
        if cmd:
            msg += " Prompt text isn't used in composite mode; use the editor's tools afterward."
    elif mode == AI_MODE:
        if not cmd:
            return ([state] + option_images(state) + ["Describe the edit you want first.",
                    gr.update(), gr.update()])
        limit = max_images()
        sel = mentioned_photos(cmd, N) or list(range(1, N + 1))
        truncated = len(sel) > limit
        sel = sel[:limit]
        refs = [(f"Photo {k}", photos[k - 1]) for k in sel[1:]]
        base, base_label = photos[sel[0] - 1], f"Photo {sel[0]}"

        def job(i):
            try:
                return ai_edit(base, cmd, base_label, refs), None
            except Exception as e:
                return None, str(e)[:200]

        with ThreadPoolExecutor(max_workers=n if using_gemini() else 1) as ex:  # free queue: one at a time
            results = list(ex.map(job, range(n)))
        for i, (out, err) in enumerate(results):
            if out is not None:
                options[i] = {"img": out,
                              "cmds": [f"AI: {cmd} (photos {', '.join(map(str, sel))})"]}
            else:
                errors.append(f"option {i + 1}: {err}")
        if not any(options):  # keep earlier options if everything failed
            return ([state] + option_images(state)
                    + ["AI edit failed. " + " | ".join(errors), gr.update(), gr.update()])
        msg = (f"AI created {sum(o is not None for o in options)} option(s) from photo(s) "
               f"{', '.join(map(str, sel))}.")
        if truncated:
            msg += (f" Limit is {limit} photo(s) per AI request; name the one(s) you want, like 'photo 2'."
                    if using_gemini() else
                    " The free AI edits one photo at a time, so only the first one named was used.")
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
    msg += extra + " Unsaved options from before were replaced."
    return ([state] + option_images(state)
            + [msg, gr.update(choices=overlay_choices(state), value="None"), gr.update()])


# ---------- STEP 2: edit one option ----------
def open_edit(slot, state):
    opt = state["options"][slot]
    if opt is None:
        return [state, gr.update(), gr.update(), None, "", f"Option {slot + 1} is empty."]
    state.update(slot=slot, orig={"img": opt["img"].copy(), "cmds": list(opt["cmds"])},
                 work=opt["img"].copy(), cmds=list(opt["cmds"]), undo=[], detections=[])
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
    if len(state["undo"]) > UNDO_LIMIT:
        state["undo"].pop(0)
    state["work"] = out
    state["cmds"].append(label)
    state["detections"] = []  # the picture changed, so old scan boxes no longer fit
    return [state, out, edit_log(state), f"Applied: {label}"]


def apply_prompt(command, overlay_choice, mode, state):
    if state["work"] is None:
        return [state, None, "", "Open an option to edit first."]
    command = (command or "").strip()
    if not command:
        return [state, state["work"], edit_log(state), "Type an edit instruction first."]
    if mode == AI_MODE:
        nums = mentioned_photos(command, len(state["photos"]))[:max_images() - 1]
        refs = [(f"Photo {k} (one of the original uploads)", state["photos"][k - 1]) for k in nums]
        return run_tool(state, f"AI: {command}",
                        lambda im: ai_edit(im, command, "the picture currently being edited", refs))
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
    state["detections"] = []
    return [state, state["work"], edit_log(state), "Undid last edit."]


def reset_edit(state):
    if state["orig"] is None:
        return [state, None, "", "Nothing to reset."]
    state.update(work=state["orig"]["img"].copy(), cmds=list(state["orig"]["cmds"]),
                 undo=[], detections=[])
    return [state, state["work"], edit_log(state), "Reset to how it was when you opened it."]


def leave_edit(state, save):
    slot = state["slot"]
    if save and slot is not None and state["work"] is not None:
        state["options"][slot] = {"img": state["work"].copy(), "cmds": list(state["cmds"])}
        msg = f"Saved changes to option {slot + 1}."
    else:
        msg = "Left editor without saving."
    state.update(slot=None, orig=None, work=None, cmds=[], undo=[], detections=[])
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


# ----- scan & remove -----
def t_scan(state):
    empty = gr.update(choices=[], value=[])
    if state["work"] is None:
        return [state, None, empty, "Open an option to edit first."]
    try:
        dets = detect_objects(state["work"])
    except Exception as e:
        return [state, None, empty, f"Scan failed: {e}"]
    state["detections"] = dets
    choices = [f"{k + 1}: {d['label']} {int(d['conf'] * 100)}%" for k, d in enumerate(dets)]
    msg = (f"Found {len(dets)} object(s). Tick the ones to remove."
           if dets else "Nothing recognized. The scan only knows everyday objects; use 'Erase a region' instead.")
    return [state, draw_detections(state["work"], dets), gr.update(choices=choices, value=[]), msg]


def t_remove_selected(selected, state):
    dets = state["detections"]
    if not dets:
        return [state, state["work"], edit_log(state), "Scan first (any edit clears the old scan)."]
    idxs = sorted({int(s.split(":")[0]) - 1 for s in (selected or [])})
    idxs = [k for k in idxs if 0 <= k < len(dets)]
    if not idxs:
        return [state, state["work"], edit_log(state), "Tick at least one detected object first."]
    names = ", ".join(dets[k]["label"] for k in idxs)

    def fn(im):
        mask = Image.new("L", im.size, 0)
        d = ImageDraw.Draw(mask)
        pad = max(4, int(max(im.size) * 0.01))
        for k in idxs:
            x0, y0, x1, y1 = dets[k]["box"]
            d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=255)
        return lama_inpaint(im, mask)
    return run_tool(state, f"remove {names} (LaMa)", fn)


def region_box(im, l, t, r, b):
    if r <= l or b <= t:
        raise ValueError("right must be greater than left, and bottom greater than top")
    w, h = im.size
    return (int(w * l / 100), int(h * t / 100), int(w * r / 100), int(h * b / 100))


def t_remove_region(l, t, r, b, state):
    def fn(im):
        mask = Image.new("L", im.size, 0)
        ImageDraw.Draw(mask).rectangle(region_box(im, l, t, r, b), fill=255)
        return lama_inpaint(im, mask)
    return run_tool(state, f"erase region ({l:g},{t:g})-({r:g},{b:g})% (LaMa)", fn)


def t_preview_region(l, t, r, b, state):
    if state["work"] is None:
        return None
    im = state["work"].copy()
    try:
        box = region_box(im, l, t, r, b)
    except ValueError:
        return im
    ImageDraw.Draw(im).rectangle(box, outline="red", width=max(3, im.width // 200))
    return im


# ---------- finalize actions ----------
def save_slot(slot, up, state):
    opt = state["options"][slot]
    if opt is None:
        gallery, text = final_render(state)
        return [state] + option_images(state) + [gallery, text, f"Option {slot + 1} is empty."]
    img, cmds, note = opt["img"].copy(), list(opt["cmds"]), ""
    if up:
        try:
            img = upscale2x(img)
            cmds.append("upscaled 2x (Real-ESRGAN)")
            note = " Upscaled 2x."
        except Exception as e:
            note = f" Upscale skipped: {e}."
    state["counter"] += 1
    state["final"].append({"id": state["counter"], "img": img, "cmds": cmds})
    state["options"][slot] = None
    gallery, text = final_render(state)
    return ([state] + option_images(state)
            + [gallery, text, f"Option {slot + 1} saved as #{state['counter']} and moved to Finalized.{note}"])


def discard_slot(slot, state):
    state["options"][slot] = None
    return [state] + option_images(state) + [f"Discarded option {slot + 1}."]


def toggle_view(mode):
    return gr.update(visible=mode == "Thumbnails"), gr.update(visible=mode == "Text")


OPEN_TAB_JS = "(img) => { if (img && img.url) { window.open(img.url, '_blank'); } }"

HELP = """**AI edit mode:** describe the change in plain English, e.g. `remove the toys from the floor`.
The free AI edits one photo at a time and may wait in a queue; set "Options to create" to 1 to save your free quota. With several photos, say which one: `remove the toys in photo 2`.
Combining photos (`put the sofa from photo 2 into photo 1`) needs a Gemini key. Results can vary, so generate again or keep editing.

**Quick commands mode** (keyword-based, not AI). Combine with commas or "then", e.g. `grayscale, blur 3, text Hello`

- **Adjust:** grayscale, brighten, darker, contrast, saturation, autocontrast, equalize, posterize, invert, solarize
- **Filter:** blur, sharpen, edges, emboss, contour, smooth, detail
- **Draw:** text <words>, box, watermark
- **Geometry:** rotate left / right / 45, mirror, flip horizontal / vertical, crop 70, resize 50%, thumbnail 256, fit 512, pad 512, expand 30
- **Layers (uses your other photos):** blend 0.5, composite 120, paste
- **Chops (vs. original):** chops difference, chops multiply, chops lighter, chops darker

In the editor, the tool tabs give you buttons and sliders for the same things, plus crop and **Scan & remove**.
"""

# ---------- compose: local people cutouts and placement ----------
SEG_PATH = os.environ.get("SEG_PATH", "models/yolov8n-seg.onnx")
_seg = None


def get_seg():
    global _seg
    if _seg is None:
        if not os.path.exists(SEG_PATH):
            raise RuntimeError(f"segmentation model not found at '{SEG_PATH}'. Run export_yolo.py "
                               "once and commit models/yolov8n-seg.onnx to GitHub.")
        import onnxruntime as ort
        _seg = ort.InferenceSession(SEG_PATH, providers=["CPUExecutionProvider"])
    return _seg


def segment_people(img, conf_thr=0.35, max_people=6, size=640):
    """Find people and their exact outlines. Returns [{'conf', 'box', 'mask'}] in image pixels."""
    sess = get_seg()
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (px, py))
    x = (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    protos = next(o for o in outs if o.ndim == 4)[0]     # (32, 160, 160)
    pred = next(o for o in outs if o.ndim == 3)[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                                     # (anchors, 4 + 80 classes + 32 mask coefs)
    ncoef = protos.shape[0]
    person = pred[:, 4]                                   # class 0 = person
    keep = person > conf_thr
    if not keep.any():
        return []
    pred, person = pred[keep], person[keep]
    coef = pred[:, pred.shape[1] - ncoef:]
    b = pred[:, :4]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    idx = nms(xyxy, person, 0.45)[:max_people]
    results = []
    for i in idx:
        m = coef[i] @ protos.reshape(ncoef, -1)
        m = (1 / (1 + np.exp(-m.reshape(160, 160)))) * 255
        mimg = Image.fromarray(m.astype(np.uint8)).resize((size, size), Image.BILINEAR)
        mimg = mimg.crop((px, py, px + nw, py + nh)).resize((w, h), Image.BILINEAR)
        x0 = float(np.clip((xyxy[i, 0] - px) / scale, 0, w))
        y0 = float(np.clip((xyxy[i, 1] - py) / scale, 0, h))
        x1 = float(np.clip((xyxy[i, 2] - px) / scale, 0, w))
        y1 = float(np.clip((xyxy[i, 3] - py) / scale, 0, h))
        boxmask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(boxmask).rectangle([int(x0), int(y0), int(x1), int(y1)], fill=255)
        mask = ImageChops.multiply(mimg, boxmask).point(lambda v: 255 if v > 127 else 0)
        results.append({"conf": float(person[i]), "box": (x0, y0, x1, y1), "mask": mask.convert("1")})
    return results


def make_cutout(img, mask):
    """RGBA crop of the person with a slightly feathered edge; None if the mask is empty."""
    bbox = mask.convert("L").getbbox()
    if bbox is None:
        return None
    alpha = mask.convert("L").filter(ImageFilter.GaussianBlur(1.2))
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba.crop(bbox)


def match_color(src, alpha, dst, strength):
    """Shift the cutout's colors toward the scene where it lands (mean and spread, limited)."""
    s = np.asarray(src, np.float32)
    a = np.asarray(alpha) > 127
    if a.sum() < 50:
        return src
    d = np.asarray(dst.convert("RGB"), np.float32).reshape(-1, 3)
    sp = s[a]
    sm, ss = sp.mean(0), sp.std(0) + 1e-3
    dm, ds = d.mean(0), d.std(0) + 1e-3
    ratio = np.clip(ds / ss, 0.7, 1.4)
    adj = (s - sm) * ratio + dm
    out = s * (1 - strength) + adj * strength
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def paste_cutout(base, cut, cx, cy, height_frac, strength):
    W, H = base.size
    th = min(H, max(8, int(H * height_frac)))
    cw = max(8, int(cut.width * th / cut.height))
    if cw > W:
        cw = W
        th = max(8, int(cut.height * cw / cut.width))
    piece = cut.resize((cw, th), Image.LANCZOS)
    x0 = min(max(int(round(cx * W - cw / 2)), 0), W - cw)
    y0 = min(max(int(round(cy * H - th / 2)), 0), H - th)
    alpha = piece.getchannel("A")
    rgb = match_color(piece.convert("RGB"), alpha, base.crop((x0, y0, x0 + cw, y0 + th)), strength)
    soft = alpha.filter(ImageFilter.GaussianBlur(max(1, cw // 80)))
    out = base.copy()
    out.paste(rgb, (x0, y0), soft)
    return out


def default_place(j, n):
    return (0.2 + 0.6 * (j + 1) / (n + 1), 0.6)


def compose_cycle(base, srcs, state, cycle):
    """One independent cycle: paste each cutout onto the base in turn, each round feeding the next."""
    rng = random.Random(1000 + cycle)
    size = state.get("size_frac", 0.45)
    out = base.copy()
    n = len(srcs)
    for j, (k, cut) in enumerate(srcs):
        cx, cy = state["placements"].get(k) or default_place(j, n)
        dx, dy = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
        scale = rng.uniform(0.93, 1.07)
        strength = rng.uniform(0.55, 0.85)
        out = paste_cutout(out, cut, cx + dx, cy + dy, size * scale, strength)
    return out


def upload_names(files):
    if not files:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    return [os.path.basename(getattr(f, "name", f)) for f in files[:MAX_PHOTOS]]


def pending_photos(state):
    return [k for k in sorted(state["people"]) if state["cutouts"].get(k) is None]


def cut_choices(state):
    return [f"Photo {k}" for k in sorted(state["cutouts"])
            if k > 1 and state["cutouts"].get(k) is not None]


def compose_sources(state):
    return [(k, state["cutouts"][k]) for k in sorted(state["cutouts"])
            if k > 1 and state["cutouts"].get(k) is not None]


def compose_ready(files, state):
    if not state.get("setup_names"):
        return False, "Click 'Find people in photos 2+' first."
    if upload_names(files) != state["setup_names"]:
        return False, "The uploads changed since you found people. Click 'Find people' again."
    if not compose_sources(state):
        return False, "No people are ready. Check the people in photos 2+ and confirm any pending ones."
    return True, ""


def person_labels(state, k):
    return [f"Person {i + 1} ({int(d['conf'] * 100)}%)" for i, d in enumerate(state["people"].get(k, []))]


def person_value(state, k):
    i = state["chosen"].get(k)
    labels = person_labels(state, k)
    return labels[i] if i is not None and i < len(labels) else None


def labeled_preview(state, k):
    return draw_detections(state["photos"][k - 1], state["people"][k])


def redraw_placements(state):
    if not state.get("photos"):
        return None
    im = state["photos"][0].copy()
    W, H = im.size
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(16, W // 30))
    for k, (fx, fy) in sorted(state["placements"].items()):
        x, y, r = fx * W, fy * H, max(8, W // 60)
        d.ellipse([x - r, y - r, x + r, y + r], outline="yellow", width=3)
        d.text((x + r + 4, y - r), str(k), fill="yellow", font=font, stroke_width=2, stroke_fill="black")
    return im


def prepare_compose(files, state):
    photos, _ = load_photos(files)
    if len(photos) < 2:
        return (state, gr.update(choices=[], value=None), None,
                gr.update(choices=[], value=None), gr.update(choices=[], value=None), None,
                "Upload at least two photos: photo 1 is the base, and photos 2+ supply people.")
    state.update(photos=photos, setup_names=upload_names(files), people={}, cutouts={},
                 chosen={}, placements={})
    errs, none_found = [], []
    for k in range(2, len(photos) + 1):
        try:
            people = segment_people(photos[k - 1])
        except Exception as e:
            errs.append(f"photo {k}: {str(e)[:120]}")
            continue
        if not people:
            none_found.append(k)
            continue
        state["people"][k] = people
        if len(people) == 1 and people[0]["conf"] >= 0.6:  # clear case: use it automatically
            state["cutouts"][k] = make_cutout(photos[k - 1], people[0]["mask"])
            state["chosen"][k] = 0
    keys = sorted(state["people"])
    pend = pending_photos(state)
    first = pend[0] if pend else (keys[0] if keys else None)
    msg = f"Found people in {len(keys)} photo(s)."
    if pend:
        msg += " Please check: " + ", ".join(f"photo {k}" for k in pend) + "."
    if none_found:
        msg += " No person found in: " + ", ".join(f"photo {k}" for k in none_found) + "."
    if errs:
        msg += " Errors: " + " | ".join(errs)
    return (state,
            gr.update(choices=[f"Photo {k}" for k in keys], value=f"Photo {first}" if first else None),
            labeled_preview(state, first) if first else None,
            gr.update(choices=person_labels(state, first) if first else [],
                      value=person_value(state, first) if first else None),
            gr.update(choices=cut_choices(state), value=None),
            redraw_placements(state), msg)


def review_pick(photo_choice, state):
    if not photo_choice or not state.get("photos"):
        return None, gr.update(choices=[], value=None)
    k = int(photo_choice.split()[1])
    return labeled_preview(state, k), gr.update(choices=person_labels(state, k),
                                                value=person_value(state, k))


def confirm_person(photo_choice, person_label, state):
    if not photo_choice or not person_label:
        return state, gr.update(), gr.update(), gr.update()
    k = int(photo_choice.split()[1])
    i = int(person_label.split()[1]) - 1
    dets = state["people"].get(k, [])
    if not (0 <= i < len(dets)):
        return state, gr.update(), gr.update(), "That person isn't in the list."
    cut = make_cutout(state["photos"][k - 1], dets[i]["mask"])
    if cut is None:
        return state, gr.update(), gr.update(), "Couldn't outline that person; try another."
    state["cutouts"][k] = cut
    state["chosen"][k] = i
    pend = pending_photos(state)
    status = f"Using person {i + 1} from photo {k}."
    if pend:
        status += " Still to check: " + ", ".join(f"photo {j}" for j in pend) + "."
    else:
        status += " All photos are checked."
    return state, gr.update(choices=cut_choices(state)), redraw_placements(state), status


def place_click(choice, state, evt: gr.SelectData):
    if not choice or not state.get("photos"):
        return state, redraw_placements(state), "Choose a cutout first."
    k = int(choice.split()[1])
    W, H = state["photos"][0].size
    x, y = evt.index
    state["placements"][k] = (min(max(x / W, 0), 1), min(max(y / H, 0), 1))
    return state, redraw_placements(state), f"Placed photo {k}'s person there. Click again to move it."


def set_size(v, state):
    state["size_frac"] = v / 100
    return state


def clear_places(state):
    state["placements"] = {}
    return state, redraw_placements(state), "Placements reset to automatic spacing."


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
                               info="AI edit understands plain English (free queue, can take a minute). Quick commands are instant keyword tools.")
        photo_files = gr.File(label=f"Upload one or more photos (up to {MAX_PHOTOS})",
                              file_count="multiple", file_types=["image"])
        with gr.Accordion("Compose people from several photos (local, no AI)", open=False):
            gr.Markdown("Photo 1 is the base. Photos 2+ supply people. Find people, check any "
                        "you're unsure of, then click where each one goes on photo 1. "
                        "Set Edit mode to Compose and press Generate.")
            prepare_btn = gr.Button("Find people in photos 2+")
            review_dd = gr.Dropdown(label="Photo to check", choices=[])
            review_img = gr.Image(type="pil", label="Labeled people (numbers match the list below)",
                                  interactive=False, height=260)
            person_radio = gr.Radio(label="Which person to use", choices=[])
            place_pick = gr.Dropdown(label="Cutout to place", choices=[])
            place_base = gr.Image(type="pil", label="Photo 1 (base): click to place the selected cutout",
                                  interactive=True, height=320)
            place_size = gr.Slider(15, 90, value=45, step=1, label="Person height, % of base photo")
            place_clear_btn = gr.Button("Reset placements")
        with gr.Row():
            num_outputs = gr.Slider(1, 3, step=1, value=3, label="Options to create")
            run_btn = gr.Button("Generate options", variant="primary")
        gr.Markdown("_Photos are shrunk automatically to save memory. In AI mode, name the photo in your prompt ('photo 2'); in Quick mode each option uses a different photo._")
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
        mode_edit = gr.Radio(EDIT_MODES, value=DEFAULT_MODE, label="Edit mode")
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

            with gr.Tab("Scan & remove"):
                gr.Markdown("**Scan** finds everyday objects on your server (no outside service). "
                            "**Remove** fills the spot from its surroundings. Any edit clears the old scan.")
                scan_btn = gr.Button("Scan image", variant="primary")
                scan_img = gr.Image(type="pil", label="Scan result", interactive=False, height=300)
                detected = gr.CheckboxGroup(label="Detected objects (tick to remove)", choices=[])
                remove_sel_btn = gr.Button("Remove ticked objects", variant="primary")
                gr.Markdown("**Erase a region** (for things the scan can't name, like toys). Positions are % of the image.")
                reg_l = gr.Slider(0, 100, value=25, step=1, label="Left edge %")
                reg_t = gr.Slider(0, 100, value=25, step=1, label="Top edge %")
                reg_r = gr.Slider(0, 100, value=75, step=1, label="Right edge %")
                reg_b = gr.Slider(0, 100, value=75, step=1, label="Bottom edge %")
                with gr.Row():
                    reg_prev_btn = gr.Button("Preview region")
                    reg_btn = gr.Button("Erase region", variant="primary")

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
    upscale_chk = gr.Checkbox(label="Upscale 2× when saving (sharper, but slower; uses Real-ESRGAN on your server)", value=False)
    view_mode = gr.Radio(["Thumbnails", "Text"], value="Thumbnails", label="View")
    final_gallery = gr.Gallery(label="Finalized", columns=3, object_fit="contain",
                               height="auto", allow_preview=True, visible=True)
    final_text = gr.Markdown("_Nothing finalized yet._", visible=False)

    # ----- wiring: compose setup -----
    prepare_btn.click(prepare_compose, [photo_files, state],
                      [state, review_dd, review_img, person_radio, place_pick, place_base, status_output])
    review_dd.change(review_pick, [review_dd, state], [review_img, person_radio])
    person_radio.change(confirm_person, [review_dd, person_radio, state],
                        [state, place_pick, place_base, status_output])
    place_base.select(place_click, [place_pick, state], [state, place_base, status_output])
    place_size.change(set_size, [place_size, state], [state])
    place_clear_btn.click(clear_places, [state], [state, place_base, status_output])

    # ----- wiring: step 1 -----
    run_btn.click(generate, [photo_files, command_input, mode_create, num_outputs, state],
                  [state] + slot_imgs + [status_output, overlay_choice, final_gallery])

    for n, (img, (b_edit, b_save, b_tab, b_discard)) in enumerate(zip(slot_imgs, slot_btns)):
        b_edit.click(lambda st, n=n: open_edit(n, st), [state],
                     [state, create_view, edit_view, edit_img, edit_history, status_output])
        b_save.click(lambda st, up, n=n: save_slot(n, up, st), [state, upscale_chk],
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

    scan_btn.click(t_scan, [state], [state, scan_img, detected, status_output])
    remove_sel_btn.click(t_remove_selected, [detected, state], E)
    reg_prev_btn.click(t_preview_region, [reg_l, reg_t, reg_r, reg_b, state], [scan_img])
    reg_btn.click(t_remove_region, [reg_l, reg_t, reg_r, reg_b, state], E)

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
