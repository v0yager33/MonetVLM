import argparse
import tempfile
import os

import gradio as gr
import torch
from inference import load_model, generate


def parse_args():
    parser = argparse.ArgumentParser(description="MonetVLM Gradio Web Demo")
    parser.add_argument("--model_dir", type=str, default="save/vlm_sft_full",
                        help="Path to the trained model directory")
    parser.add_argument("--port", type=int, default=7891, help="Server port")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


args = parse_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing MonetVLM from {args.model_dir}...")
try:
    model, processor, tokenizer = load_model(args.model_dir, DEVICE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Please make sure you have trained and saved the model first.")
    model, processor, tokenizer = None, None, None


def chat_interface(image_input, text_input):
    if model is None:
        return "Model not loaded. Please check the model path."
    if image_input is None:
        return "Please upload an image."
    if not text_input or not text_input.strip():
        text_input = "Describe this image."

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_input.convert("RGB").save(tmp.name)
        tmp_path = tmp.name

    try:
        response = generate(
            model, processor, tokenizer, text_input,
            image_path=tmp_path, max_new_tokens=args.max_new_tokens, device=DEVICE,
        )
    except Exception as e:
        response = f"Error during generation: {e}"
    finally:
        os.unlink(tmp_path)

    return response


with gr.Blocks(title="MonetVLM") as demo:
    gr.Markdown("## MonetVLM (Qwen3-1.7B + SigLIP2) Web Demo")

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Upload Image")
            text_in = gr.Textbox(label="Prompt", placeholder="Describe this image.")
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            text_out = gr.Textbox(label="Assistant Response", lines=10, interactive=False)

    submit_btn.click(fn=chat_interface, inputs=[img_in, text_in], outputs=[text_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)
