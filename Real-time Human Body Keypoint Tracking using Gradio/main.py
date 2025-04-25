import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')  # or yolov8s-pose.pt for better accuracy
model.to(device)
def process_frame(frame):
    """Run YOLO pose estimation on a frame."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model(frame_bgr, verbose=False)
    annotated_frame = results[0].plot()
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

def process_video():
    """Stream webcam frames and process them for pose estimation."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip input frame horizontally to counter browser mirroring effect
        flipped_frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        # Process the *flipped* frame for pose estimation, so output matches.
        processed_frame = process_frame(frame_rgb)

        yield frame_rgb, processed_frame

    cap.release()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Real-time Pose Estimation with YOLOv8")

    with gr.Row():
        input_video = gr.Image(label="Webcam Input", streaming=True)
        output_video = gr.Image(label="Pose Output", streaming=True)

    demo.load(
        fn=process_video,
        inputs=None,
        outputs=[input_video, output_video]
    )

if __name__ == "__main__":
    demo.launch()
