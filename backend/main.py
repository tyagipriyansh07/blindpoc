from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import os
from logic import decide_action
from utils import call_groq_llm

app = FastAPI()

# Allow Streamlit to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("yolo/yolov8n.pt")

@app.post("/analyze")
async def analyze(image: UploadFile, user_text: str = Form("")):
    # Read image
    img_bytes = await image.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Object detection
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "cls": cls,
            "conf": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # Rule-based logic
    logic_output = decide_action(detections)

    # LLM reasoning
    llm_reply = call_groq_llm(
        user_text=user_text,
        scene=detections,
        rule_output=logic_output
    )

    return {
        "detections": detections,
        "rule_output": logic_output,
        "assistant_reply": llm_reply
    }
