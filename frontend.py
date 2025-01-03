import streamlit as st
from PIL import Image
import cv2
import threading
import shutil
import os
import numpy as np
import queue
from UtilsFiles.Fight_utils import loadModel, predict_on_video, start_streaming
from emailfinal import send_email_all_files


# Initialize session state variables
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = threading.Event()
if "thread" not in st.session_state:
    st.session_state.thread = None
frame_queue = queue.Queue()

# stop_streaming_flag = threading.Event()
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
        if os.path.exists("Output_video_folder2/") and os.path.isdir("Output_video_folder2/"):
            shutil.rmtree("Output_video_folder2/")
            os.makedirs("Output_video_folder2/")
        shutil.copy2("Output_video_folder/Output_video.mp4", "Output_video_folder2/Output_video.mp4")
        st.video("Output_video_folder2/Output_video.mp4")
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
        if st.button("Start Fight Detection"):
            if st.session_state.thread is None or not st.session_state.thread.is_alive():
                st.session_state.stop_flag.clear()
                # Run start_streaming in a separate thread
                st.session_state.thread = threading.Thread(target=start_streaming, args=(model, rtsp_url, st.session_state.stop_flag, frame_queue))
                # rtsp_thread.daemon = True  # Ensures it closes with the app
                st.session_state.thread.start()
                st.success("RTSP streaming started. Observing for Fight Detection!")

        # Stop RTSP streaming
        if st.button("Stop RTSP Streaming"):
            # st.success(f"Bedore code is stoped {st.session_state.stop_flag}")
            st.session_state.stop_flag.set()
            if st.session_state.thread is not None:
                st.session_state.thread.join()  # Wait for the thread to exit
                st.session_state.thread = None  # Reset thread state
            # st.success(f"Code stoped bcos of {st.session_state.stop_flag}")
            if user_email:
                st.success(f"Processing frames, alerts will be sent to {user_email}")
                send_email_all_files(user_email, "Output_video_folder/")
                st.success(f"Alerts sent to {user_email}")
                if os.path.exists("Output_video_folder/") and os.path.isdir("Output_video_folder/"):
                    shutil.rmtree("Output_video_folder/")

        # Display Frames
        st.markdown("### RTSP Stream")
        frame_placeholder = st.empty()

        while st.session_state.thread and not st.session_state.stop_flag.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)
                frame_placeholder.image(frame_image, caption="Streaming Frame", use_container_width=True)


# Webcam Input Option
elif input_method == "Webcam":
    # st.write("Initializing webcam...")
    if st.button("Start Webcam"):
        st.write("Accessing webcam feed...")
        if st.session_state.thread is None or not st.session_state.thread.is_alive():
            st.session_state.stop_flag.clear()
            # Run start_streaming in a separate thread
            st.session_state.thread = threading.Thread(target=start_streaming, args=(model, 0, st.session_state.stop_flag, frame_queue))
            # rtsp_thread.daemon = True  # Ensures it closes with the app
            st.session_state.thread.start()
            st.success("Webcam streaming started. Observing for Fight Detection!")

    # Stop RTSP streaming
    if st.button("Stop Webcam Streaming"):
        # st.success(f"Bedore code is stoped {st.session_state.stop_flag}")
        st.session_state.stop_flag.set()
        if st.session_state.thread is not None:
            st.session_state.thread.join()  # Wait for the thread to exit
            st.session_state.thread = None  # Reset thread state
        # st.success(f"Code stoped bcos of {st.session_state.stop_flag}")
        if user_email:
            st.success(f"Processing frames, alerts will be sent to {user_email}")
            send_email_all_files(user_email, "Output_video_folder/")
            st.success(f"Alerts sent to {user_email}")
            if os.path.exists("Output_video_folder/") and os.path.isdir("Output_video_folder/"):
                shutil.rmtree("Output_video_folder/")

    # Display Frames
    st.markdown("### Webcam Stream")
    frame_placeholder = st.empty()

    while st.session_state.thread and not st.session_state.stop_flag.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_image, caption="Streaming Frame", use_container_width=True)




# # Results Section
# st.markdown("---")
# st.header("Detection Results")
# st.write("Results will appear here after processing.")
# st.empty()  # Placeholder for live updates
