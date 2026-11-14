import gradio as gr
from PIL import Image

def process_image(image, command):
    """
    Placeholder function for FuelStrictStudio image editing and text command parsing.
    Connect your backend logic, AI models, or API keys here.
    """
    if image is None:
        return None, "Please upload an image first."
    
    # Example placeholder transformation: converts image to grayscale if command matches
    if "grayscale" in command.lower():
        processed_image = image.convert("L").convert("RGB")
    else:
        # Default: return the original image back with a confirmation message
        processed_image = image
        
    response_message = f"Successfully parsed command: '{command}'"
    return processed_image, response_message

# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⛽ FuelStrictStudio")
    gr.Markdown("Secure server-backed image editing studio with precise text command parsing.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Source Image")
            command_input = gr.Textbox(label="Text Command", placeholder="Enter your editing instruction...")
            submit_btn = gr.Button("Run Command", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(type="pil", label="Processed Output")
            output_status = gr.Textbox(label="Status / Log", interactive=False)
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, command_input],
        outputs=[output_image, output_status]
    )

if __name__ == "__main__":
    # Render binds dynamically to the PORT environment variable
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
