# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from ultralytics import YOLO
# import numpy as np
# import cv2
# import os
# from backend.logic import decide_action
# from backend.utils import call_groq_llm

# app = FastAPI()

# # Allow Streamlit to call backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load YOLO model
# model = YOLO("yolo/yolov8n.pt")

# @app.post("/analyze")
# async def analyze(image: UploadFile, user_text: str = Form("")):
#     # Read image
#     img_bytes = await image.read()
#     frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

#     # Object detection
#     results = model(frame)[0]
#     detections = []

#     for box in results.boxes:
#         cls = model.names[int(box.cls[0])]
#         conf = float(box.conf[0])
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         detections.append({
#             "cls": cls,
#             "conf": conf,
#             "bbox": [x1, y1, x2, y2]
#         })

#     # Rule-based logic
#     logic_output = decide_action(detections)

#     # LLM reasoning
#     llm_reply = call_groq_llm(
#         user_text=user_text,
#         scene=detections,
#         rule_output=logic_output
#     )

#     return {
#         "detections": detections,
#         "rule_output": logic_output,
#         "assistant_reply": llm_reply
#     }


# @app.post("/analyze_video")
# async def analyze_video(image: UploadFile = File(...)):
#     contents = await image.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     results = model(frame, verbose=False)
#     detections = [r for r in results[0].boxes.data.tolist()]

#     rule_output = decide_action(detections)

#     # Let LLM speak
#     assistant_reply = call_groq_llm(
#         "Describe the scene.",
#         detections,
#         rule_output
#     )

#     return {
#         "detections": detections,
#         "rule_output": rule_output,
#         "assistant_reply": assistant_reply
#     }



# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import numpy as np
import cv2
import os

from backend.logic import decide_action
from backend.utils import call_groq_llm

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audio mount
if not os.path.exists("audio"):
    os.makedirs("audio")

app.mount("/audio", StaticFiles(directory="audio"), name="audio")

# Load YOLO
model = YOLO("yolo/yolov8n.pt")

@app.post("/analyze_video")
async def analyze_video(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False)[0]

    h, w, _ = frame.shape

    detections = []

    # Extract normalized size & position
    for box in results.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        rel_w = (x2 - x1) / w
        rel_h = (y2 - y1) / h

        detections.append({
            "cls": cls,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            "cx": cx,
            "cy": cy,
            "rel_w": rel_w,
            "rel_h": rel_h
        })

    # Apply smart assistive logic
    rule_output = decide_action(detections)

    # LLM summary
    assistant_reply = call_groq_llm(
        "Provide short guidance for blind user.",
        detections,
        rule_output
    )

    return {
        "detections": detections,
        "assistant_reply": assistant_reply,
        "rule_output": rule_output
    }


