import streamlit as st
import cv2
import numpy as np
import tempfile
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("best_model.keras")

# Define class labels (update these based on your model training)
class_labels = ['drive', 'pull shot', 'sweep', 'cut shot']

st.title("🏏 Cricket Shot Identification")

# Upload video
video_file = st.file_uploader("Upload a cricket video", type=["mp4", "mov", "avi"])

if video_file:
    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Get input second for frame capture
    sec = st.number_input("Enter time (in seconds) to capture frame", min_value=0, step=1)
    
    if st.button("Predict Shot"):
        # Open video
        cap = cv2.VideoCapture(tfile.name)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * sec)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            # Preprocess frame for model
            frame_resized = cv2.resize(frame, (224, 224))  # adjust if your model input size is different
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb / 255.0
            input_array = np.expand_dims(frame_normalized, axis=0)
            
            # Predict
            predictions = model.predict(input_array)
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions)
            
            st.image(frame_rgb, caption=f"Captured frame at {sec} sec", channels="RGB")
            st.success(f"Predicted Shot: {predicted_class} ({confidence*100:.2f}% confidence)")
        
        else:
            st.error("Failed to capture frame. Please check the second you entered.")

