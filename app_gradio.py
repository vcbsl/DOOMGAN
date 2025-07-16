import gradio as gr
import torch
import yaml
import os

# Local project imports (reusing the exact same backend logic)
from apps.utils import load_all_models
from apps.gan_morpher import morph_images_with_gan
from apps.classical_morpher import ocular_morph_classical, predict_landmarks_for_classical

# --- 1. Load Configuration and Models (once, when the script starts) ---
print("--- Initializing Gradio App: Loading Config and Models ---")
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NOTE: Change the default epoch here if needed
models = load_all_models(config, epoch=450, device=device)
print("--- Initialization Complete ---")

# --- 2. Define the processing functions that Gradio will call ---

def perform_classical_morph(image1, image2):
    """Gradio-compatible function for the classical morphing method."""
    if image1 is None or image2 is None:
        raise gr.Error("Please upload both images for classical morphing.")
    
    print("Performing classical morph...")
    landmarks1 = predict_landmarks_for_classical(image1, models['landmark_predictor'], device)
    landmarks2 = predict_landmarks_for_classical(image2, models['landmark_predictor'], device)
    morphed_image = ocular_morph_classical(image1, image2, landmarks1, landmarks2)
    print("Classical morph complete.")
    return morphed_image

def perform_gan_morph(image1, image2, alpha):
    """Gradio-compatible function for the GAN morphing method."""
    if image1 is None or image2 is None:
        raise gr.Error("Please upload both images for GAN morphing.")
    
    print(f"Performing GAN morph with alpha={alpha}...")
    morphed_image = morph_images_with_gan(image1, image2, config, device, models, alpha)
    print("GAN morph complete.")
    return morphed_image

# --- 3. Build the Gradio Interface using Blocks and Tabs ---

with gr.Blocks(theme=gr.themes.Soft(), title="Ocular Morphing") as demo:
    gr.Markdown("# High Fidelity Ocular Image Morphing")
    gr.Markdown("An interactive demonstration of the IJCB-accepted **DOOMGAN** project. Use the tabs below to switch between our GAN-based method and a classical approach.")

    with gr.Tabs():
        with gr.TabItem("GAN Morph (DOOMGAN)"):
            with gr.Row():
                with gr.Column():
                    gan_img1 = gr.Image(type="pil", label="Source Image 1")
                    gan_img2 = gr.Image(type="pil", label="Source Image 2")
                gan_output = gr.Image(type="pil", label="Morphed Result")
            
            alpha_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                label="Interpolation Factor (Image 1 <-> Image 2)"
            )
            gan_button = gr.Button("Generate GAN Morph", variant="primary")

        with gr.TabItem("Classical Morph (LM)"):
            with gr.Row():
                classical_img1 = gr.Image(type="pil", label="Source Image 1")
                classical_img2 = gr.Image(type="pil", label="Source Image 2")
                classical_output = gr.Image(type="pil", label="Morphed Result")
            
            classical_button = gr.Button("Generate Classical Morph", variant="primary")

    # --- 4. Connect the UI components to the functions ---
    gan_button.click(
        fn=perform_gan_morph,
        inputs=[gan_img1, gan_img2, alpha_slider],
        outputs=[gan_output]
    )
    
    classical_button.click(
        fn=perform_classical_morph,
        inputs=[classical_img1, classical_img2],
        outputs=[classical_output]
    )

# --- 5. Launch the application ---
if __name__ == "__main__":
    demo.launch()