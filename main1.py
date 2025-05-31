from ultralytics import YOLO
from PIL import Image
import streamlit as st
import os

# Load the YOLO model
model = YOLO("epoch-11.pt")

# Streamlit App Title and Description
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection System")
st.markdown("""
Welcome to the Brain Tumor Detection System!  
Upload an **image** or **video**, and the system will analyze it using a YOLO model trained for brain tumor detection.  
*Supported formats: JPG, JPEG, PNG, MP4*
""")

# File Uploader
uploaded_file = st.file_uploader("📤 Upload your file below:", type=["jpg", "jpeg", "png", "mp4"])

# Process Uploaded File
if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

    # Handle Image Files
    if file_extension in [".jpg", ".jpeg", ".png"]:
        st.markdown("### 📷 Uploaded Image Preview")
        try:
            # Open and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True, channels="RGB")

            # Run the model prediction
            st.markdown("### 🔍 Detection in Progress...")
            results = model.predict(source=image)

            # Display the detection results
            st.image(results[0].plot(), caption="Detection Results", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

    # Handle Video Files
    elif file_extension in [".mp4"]:
        st.markdown("### 🎥 Uploaded Video Preview")
        try:
            # Save and display the uploaded video
            with open("uploaded_video.mp4", "wb") as f:
                f.write(uploaded_file.read())
            st.video("uploaded_video.mp4", format="video/mp4")

            # Run the model prediction
            st.markdown("### 🔍 Detection in Progress...")
            results = model.predict(source="uploaded_video.mp4")

            # Display the results (if applicable for video processing)
            st.markdown("### ✅ Detection Completed!")
            st.success("The video has been successfully analyzed. Check the output.")
        except Exception as e:
            st.error(f"❌ Error processing video: {e}")

    # Unsupported File Formats
    else:
        st.error("❌ Unsupported file format. Please upload a valid image or video.")
else:
    st.info("📤 Please upload an image or video file to begin the detection process.")
