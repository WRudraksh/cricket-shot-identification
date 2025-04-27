# Cricket Shot Recognition from Videos 🎯🏏

This project focuses on identifying different cricket shots from a video input using deep learning and computer vision techniques.  
Shots classified: **Sweep**, **Flick**, **Pull**, **Drive**.

## 📚 Project Overview

- **Dataset**: Cricket shot images (Sweep, Flick, Pull, Drive) from Kaggle.
- **Model**: Transfer Learning with **MobileNetV2**.
- **App**: A simple **Streamlit** app that:
  - Takes a video input from the user.
  - Accepts a time (in seconds) to capture a frame.
  - Predicts the shot based on the extracted frame.

## 🚀 How It Works

1. Upload your cricket video.
2. Enter the specific time (seconds) to capture a frame.
3. The trained model identifies and classifies the shot.

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
