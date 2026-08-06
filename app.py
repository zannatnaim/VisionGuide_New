import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
from utils.speaker import speak, describe_scene

st.set_page_config(page_title="VisionGuide", layout="wide")
st.title("🦯 VisionGuide — AI Voice Assistant")
st.markdown("### Smart Guide for Visually Impaired")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
    voice_rate = st.slider("Voice Speed", 100, 200, 150)
    
    st.header("Info")
    st.write("Model: YOLOv8n")
    st.write("Classes: 80 (COCO)")

tab1, tab2 = st.tabs(["Webcam", "Upload Image"])

with tab1:
    st.warning("Click 'Start Camera' to begin")
    
    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera.")
        else:
            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame)
                annotated = results[0].plot()
                
                detections = []
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf > confidence_threshold:
                            cls = int(box.cls[0])
                            label = model.names[cls]
                            detections.append({'label': label, 'confidence': conf})
                
                if detections:
                    scene_desc = describe_scene(detections)
                    stframe.image(annotated, channels="BGR", use_container_width=True)
                    speak(scene_desc, voice_rate)
                else:
                    stframe.image(annotated, channels="BGR", use_container_width=True)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()

with tab2:
    uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        results = model(image)
        annotated = results[0].plot()
        
        with col2:
            st.image(annotated, channels="BGR", caption="Detected Objects", use_container_width=True)
        
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > confidence_threshold:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    detections.append({'label': label, 'confidence': conf})
        
        if detections:
            scene_desc = describe_scene(detections)
            st.success(f"✅ {scene_desc}")
            speak(scene_desc, voice_rate)
        else:
            st.warning("No objects detected with current confidence threshold.")

st.markdown("---")
st.caption("Powered by YOLOv8 | Built for Accessibility")