import streamlit as st
import torch
import pandas as pd
import os
import yaml

# Local project imports
from apps.utils import load_all_models, load_image
from apps.gan_morpher import morph_images_with_gan
from apps.classical_morpher import ocular_morph_classical, predict_landmarks_for_classical

def main():
    st.set_page_config(layout="wide")
    st.title("High Fidelity Ocular Image Morphing in the Visible Spectrum")

    # --- Load Configuration and Models (Cached) ---
    @st.cache_resource
    def setup_app(config_path='config/config.yaml', epoch=450):
        print("--- Initializing App: Loading Config and Models ---")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = load_all_models(config, epoch, device)
        print("--- Initialization Complete ---")
        return config, device, models

    try:
        # NOTE: Change the default epoch here if needed
        config, device, models = setup_app(epoch=450) 
    except Exception as e:
        st.error(f"Fatal Error during application setup: {e}")
        st.error("Please ensure all model weights and paths in 'config/config.yaml' are correct and all required files (like 'models/landmark_predictor.py') are present.")
        return

    # --- UI and Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Abstract", "Classical Morph (LM)", "GAN Morph (DOOMGAN)", "Survey"],
    )

    if page == "Home":
        st.header("Welcome to the Ocular Morphing Demonstrator")
        st.write("""
            This application showcases the capabilities of our IJCB-accepted project, **DOOMGAN**. 
            You can compare our state-of-the-art Generative Adversarial Network (GAN) morphing approach with a classical landmark-based computer vision technique.
            
            Use the sidebar to navigate between the different methods.
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.image("assets/eye_morphing_comparison.png", caption="DOOMGAN Morphing Result", use_container_width=True)
        with col2:
            st.image("assets/Intro GAN.jpg", caption="DOOMGAN Model Architecture", use_container_width=True)

    elif page == "Abstract":
        st.header("Abstract")
        st.image("assets/Final_Morph_Compare.jpg", caption="Ocular Morphing Teaser", use_container_width=True)
        st.write("""
            Ocular morphing is a technique that allows for the blending and transformation of eye images, along with the surrounding periocular region. 
            This process utilizes advanced computer vision techniques, including landmark detection and generative models, to create smooth transitions between different ocular appearances.
            Explore the various methods available in this application, including landmark-based morphing and our novel GAN approach.
        """)
    
    elif page == "Classical Morph (LM)":
        st.header("Classical Morphing via Delaunay Triangulation")
        st.info("This method uses a pre-trained model to find eye landmarks, then warps triangles between the two images to create the morph.")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file1 = st.file_uploader("Upload Image 1", type=['png', 'jpg', 'jpeg'], key="lm_img1")
        with col2:
            uploaded_file2 = st.file_uploader("Upload Image 2", type=['png', 'jpg', 'jpeg'], key="lm_img2")

        if uploaded_file1 and uploaded_file2:
            img1 = load_image(uploaded_file1)
            img2 = load_image(uploaded_file2)

            st.write("Original Images:")
            c1, c2 = st.columns(2)
            c1.image(img1, caption="Image 1", use_container_width=True)
            c2.image(img2, caption="Image 2", use_container_width=True)
            
            if st.button("Generate Classical Morph"):
                with st.spinner("Predicting landmarks and morphing..."):
                    landmarks1 = predict_landmarks_for_classical(img1, models['landmark_predictor'], device)
                    landmarks2 = predict_landmarks_for_classical(img2, models['landmark_predictor'], device)
                    morphed_image = ocular_morph_classical(img1, img2, landmarks1, landmarks2)
                    st.subheader("Morphed Result")
                    st.image(morphed_image, caption="Classical Morph", use_container_width=True)
    
    elif page == "GAN Morph (DOOMGAN)":
        st.header("State-of-the-Art Morphing with DOOMGAN")
        st.info("This method uses our trained Encoder-Generator architecture to blend the images in a learned latent space for higher fidelity results.")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file1 = st.file_uploader("Upload Image 1", type=['png', 'jpg', 'jpeg'], key="gan_img1")
        with col2:
            uploaded_file2 = st.file_uploader("Upload Image 2", type=['png', 'jpg', 'jpeg'], key="gan_img2")

        if uploaded_file1 and uploaded_file2:
            img1 = load_image(uploaded_file1)
            img2 = load_image(uploaded_file2)

            st.write("Original Images:")
            c1, c2 = st.columns(2)
            c1.image(img1, caption="Image 1", use_container_width=True)
            c2.image(img2, caption="Image 2", use_container_width=True)

            alpha = st.slider(
                "Interpolation Factor (Image 1 <-> Image 2)",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="0.0 = 100% Image 1, 1.0 = 100% Image 2, 0.5 = 50/50 blend."
            )

            if st.button("Generate GAN Morph"):
                with st.spinner("Encoding images and generating morph..."):
                    morphed_image = morph_images_with_gan(img1, img2, config, device, models, alpha)
                    st.subheader("Morphed Result")
                    st.image(morphed_image, caption=f"GAN Morph (Alpha: {alpha:.2f})", use_container_width=True)

    elif page == "Survey":
        st.header("Survey Form")
        st.write("Please provide your feedback on this application:")
        
        with st.form(key='feedback_form'):
            name = st.text_input("Name (Optional)")
            feedback = st.text_area("Feedback", "I liked...", height=150)
            submit_button = st.form_submit_button("Submit Feedback")
            
            if submit_button:
                if feedback and feedback != "I liked...":
                    logs_dir = "User Logs"
                    os.makedirs(logs_dir, exist_ok=True)
                    feedback_file = os.path.join(logs_dir, "user_feedback.csv")
                    
                    df = pd.DataFrame([{"Name": name, "Feedback": feedback}])
                    df.to_csv(feedback_file, mode='a', header=not os.path.exists(feedback_file), index=False)
                    
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Please provide some feedback before submitting.")

if __name__ == "__main__":
    main()