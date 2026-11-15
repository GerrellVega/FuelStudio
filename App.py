import os
import random
import gradio as gr
from PIL import Image, ImageOps, ImageEnhance


def make_variant(base, command, i):
    img = base.copy()
    cmd = (command or "").lower()
    if "grayscale" in cmd or "black and white" in cmd:
        return ImageOps.grayscale(img).convert("RGB")
    if "bright" in cmd:
        return ImageEnhance.Brightness(img).enhance(1.3 + i * 0.1)
    if "contrast" in cmd:
        return ImageEnhance.Contrast(img).enhance(1.2 + i * 0.1)
    if i == 1:
        return ImageOps.invert(img)
    if i == 2:
        return ImageOps.solarize(img, threshold=128)
    return img


def process_pipeline(image, command, num_outputs, history_index):
    if image is None:
        return None, None, None, "Please upload a source image first."
    base = image.convert("RGB")
    outputs = [make_variant(base, command, i) for i in range(int(num_outputs))]
    outputs += [None] * (3 - len(outputs))
    msg = (f"Generated {int(num_outputs)} variation(s) for '{command}' "
           f"(layer index: {history_index})")
    return outputs[0], outputs[1], outputs[2], msg


def regen_slot(slot, image, command):
    if image is None:
        return None, "Upload a source image first."
    # random offset so regenerating gives a different strength
    variant = make_variant(image.convert("RGB"), command, slot + random.randint(0, 3))
    return variant, f"Regenerated slot {slot + 1}"


def delete_slot(slot):
    return None, f"Deleted slot {slot + 1}"


def approve_slot(slot, img):
    if img is None:
        return f"Slot {slot + 1} is empty."
    return f"Approved slot {slot + 1}"


def edit_slot(slot, img, command):
    if img is None:
        return None, f"Slot {slot + 1} is empty."
    return make_variant(img.convert("RGB"), command, 0), f"Edited slot {slot + 1} with '{command}'"


def step_layer(direction, current_idx):
    new_idx = max(0, current_idx + direction)
    return new_idx, f"Switched to history layer index {new_idx}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⛽ FuelStrictStudio: Multi-Output & Layered Studio")
    gr.Markdown("Server-backed workspace with 1–3 parallel outputs, slot management, and history navigation.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Source Image")
            command_input = gr.Textbox(
                label="Text Command / Editing Instructions",
                placeholder="e.g., 'grayscale', 'brighten', 'increase contrast'...")
            with gr.Row():
                num_outputs_slider = gr.Slider(1, 3, step=1, value=3, label="Proposed Outputs")
                run_btn = gr.Button("Generate Options", variant="primary")

            gr.Markdown("### Layer History Navigation")
            with gr.Row():
                prev_layer_btn = gr.Button("◀ Previous Layer")
                next_layer_btn = gr.Button("Next Layer ▶")
            history_state = gr.State(value=0)
            status_output = gr.Textbox(label="Studio Status / Log", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### Proposed Output Slots (1–3)")
            slot_imgs = []
            slot_btns = []
            for n in range(3):
                with gr.Group():
                    gr.Markdown(f"**Output Slot {n + 1}**")
                    img = gr.Image(type="pil", label=f"Variation {n + 1}")
                    with gr.Row():
                        approve = gr.Button("Approve", variant="secondary")
                        edit = gr.Button("Edit + Text")
                        delete = gr.Button("Delete", variant="stop")
                        regen = gr.Button("Regenerate Slot")
                slot_imgs.append(img)
                slot_btns.append((approve, edit, delete, regen))

    # Generation
    run_btn.click(
        fn=process_pipeline,
        inputs=[input_image, command_input, num_outputs_slider, history_state],
        outputs=slot_imgs + [status_output],
    )

    # Per-slot actions (n=n binds the loop variable)
    for n, (img, (approve, edit, delete, regen)) in enumerate(zip(slot_imgs, slot_btns)):
        approve.click(lambda im, n=n: approve_slot(n, im), [img], [status_output])
        edit.click(lambda im, cmd, n=n: edit_slot(n, im, cmd),
                   [img, command_input], [img, status_output])
        delete.click(lambda n=n: delete_slot(n), None, [img, status_output])
        regen.click(lambda src, cmd, n=n: regen_slot(n, src, cmd),
                    [input_image, command_input], [img, status_output])

    # Layer navigation
    prev_layer_btn.click(lambda idx: step_layer(-1, idx), [history_state],
                         [history_state, status_output])
    next_layer_btn.click(lambda idx: step_layer(1, idx), [history_state],
                         [history_state, status_output])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
