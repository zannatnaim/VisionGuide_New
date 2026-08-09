# 🦯 VisionGuide — AI Voice Assistant for the Visually Impaired

VisionGuide is a real-time object detection and voice-narration web app built to help visually impaired users understand their surroundings. It uses **YOLOv8** to detect objects through a live camera feed, a snapshot camera, or an uploaded image — and then **speaks a natural-language description** of the scene aloud.

## 🔗 Live Demo

👉 [Try VisionGuide on Streamlit Cloud](https://visionguidenew-ai.streamlit.app/)

## ✨ Features

- **🎥 Live Camera Detection** — Real-time object detection through your webcam using WebRTC, with continuous voice narration of the scene.
- **📸 Snapshot Camera** — Take a single photo and get instant detection + spoken description.
- **📤 Image Upload** — Upload any image (JPG/PNG) and get annotated detection results with audio narration.
- **🗣️ Natural Voice Descriptions** — Converts detected objects into human-friendly sentences using Google Text-to-Speech (gTTS).
- **⚙️ Adjustable Settings** — Sidebar controls for confidence threshold, voice speed, and voice announcement interval.
- **🌐 80 Object Classes** — Powered by the YOLOv8n model trained on the COCO dataset.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Object Detection | YOLOv8 (Ultralytics) |
| Live Video Streaming | streamlit-webrtc |
| Text-to-Speech | gTTS |
| Image Processing | OpenCV (headless), Pillow |
| Language | Python 3.11 |

## 🚀 Run Locally

```bash
git clone https://github.com/zannatnaim/VisionGuide_New.git
cd VisionGuide_New
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 👤 About the Developer

**MD JANNATUL NAIM**

- GitHub: [@zannatnaim](https://github.com/zannatnaim)

---
*Built with ❤️ using YOLOv8 and Streamlit — for a more accessible world.*