import streamlit as st
from PIL import Image
import cv2
import shutil
import os
import numpy as np
from UtilsFiles.Fight_utils import loadModel, predict_on_video, start_streaming
from emailfinal import send_email_all_files

model = loadModel("/Users/anushkatyagi/Desktop/git_folder/fight_detection/Models/model_16_m3_0.8888.pth")
# --- Streamlit UI ---
st.set_page_config(page_title="Fight Detection Dashboard", layout="wide")

# Header Section
st.title("Fight Detection System")
st.markdown("### Real-Time Fight Detection and Analysis")
st.write("Identify potential fights in videos or real-time streams efficiently.")

# Sidebar Menu
st.sidebar.title("Choose Input Method")
input_method = st.sidebar.radio(
    "Select the source for analysis:",
    ("Saved Video", "RTSP Server", "Webcam")
)
# Email Input for Alerts
user_email = st.text_input("Enter your email to receive real-time alerts:")

# File Upload for Saved Video Option
if input_method == "Saved Video":
    uploaded_file = st.file_uploader("Upload a video for conflict detection:", type=["mp4", "avi", "mkv"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.write("Processing uploaded video...")
        
        # Save the uploaded video to a temporary file
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Video successfully uploaded!")

        # Use OpenCV or any other library to process the saved video
        video_path = "uploaded_video.mp4"
        # cap = cv2.VideoCapture(video_path)
        predict_on_video(video_path, model)
        # Add backend logic here to process the uploaded video
        if user_email:
            st.success(f"Alerts will be sent to {user_email} upon fight detection.")
            send_email_all_files(user_email, "Output_video_folder/")
            st.success(f"Alerts sent to {user_email}")
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists("Output_video_folder/") and os.path.isdir("Output_video_folder/"):
            shutil.rmtree("Output_video_folder/")

# RTSP Server Input Option
elif input_method == "RTSP Server":
    rtsp_url = st.text_input("Enter the RTSP server URL:")
    if rtsp_url:
        st.write(f"Streaming from RTSP server: {rtsp_url}")
        st.write("(Backend processing logic to be added)")
        # Placeholder for backend integration
        if user_email:
            st.success(f"Alerts will be sent to {user_email} upon fight detection.")
            # Call send_email_alert() when a fight is detected (use the real folder path)

# Webcam Input Option
elif input_method == "Webcam":
    st.write("Initializing webcam...")
    if st.button("Start Webcam"):
        st.write("Accessing webcam feed...")
        # Add backend logic to process webcam feed here
        if user_email:
            st.success(f"Alerts will be sent to {user_email} upon fight detection.")
            # Call send_email_alert() when a fight is detected (use the real folder path)


# # Results Section
# st.markdown("---")
# st.header("Detection Results")
# st.write("Results will appear here after processing.")
# st.empty()  # Placeholder for live updates
