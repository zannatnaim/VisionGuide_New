import time
import queue
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from utils.speaker import speak, describe_scene

st.set_page_config(page_title="VisionGuide", layout="wide")
st.title("🦯 VisionGuide — AI Voice Assistant")
st.markdown("### Smart Guide for Visually Impaired")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
    voice_rate = st.slider("Voice Speed", 100, 200, 150)
    announce_interval = st.slider("Voice Announce Interval (seconds)", 2, 10, 4)

    st.header("Info")
    st.write("Model: YOLOv8n")
    st.write("Classes: 80 (COCO)")

tab1, tab2 = st.tabs(["Live Camera", "Upload Image"])


def run_detection(pil_image, threshold):
    results = model(pil_image)
    annotated = results[0].plot()

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > threshold:
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append({'label': label, 'confidence': conf})
    return annotated, detections


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.5
        self.result_queue = queue.Queue(maxsize=1)
        self.frame_count = 0
        self.process_every_n_frames = 5
        self.last_annotated = None

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        try:
            if self.frame_count % self.process_every_n_frames == 0:
                results = model(img, verbose=False, imgsz=320)
                annotated = results[0].plot()
                self.last_annotated = annotated

                detections = []
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf > self.confidence_threshold:
                            cls = int(box.cls[0])
                            label = model.names[cls]
                            detections.append({'label': label, 'confidence': conf})

                scene_desc = describe_scene(detections)

                if not self.result_queue.full():
                    try:
                        self.result_queue.put_nowait(scene_desc)
                    except queue.Full:
                        pass
            else:
                annotated = self.last_annotated if self.last_annotated is not None else img

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
            return av.VideoFrame.from_ndarray(img, format="bgr24")


with tab1:
    st.info("Allow camera access in your browser to start live detection.")

    webrtc_ctx = webrtc_streamer(
        key="visionguide-live",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 320},
                "height": {"ideal": 240},
                "frameRate": {"ideal": 10, "max": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    description_placeholder = st.empty()
    audio_placeholder = st.empty()

    if "last_spoken_text" not in st.session_state:
        st.session_state.last_spoken_text = ""
    if "last_spoken_time" not in st.session_state:
        st.session_state.last_spoken_time = 0.0

    if webrtc_ctx.state.playing:
        while True:
            if webrtc_ctx.video_processor:
                try:
                    result = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None

                if result:
                    description_placeholder.markdown(f"**Scene:** {result}")

                    now = time.time()
                    should_speak = (
                        result != st.session_state.last_spoken_text
                        and (now - st.session_state.last_spoken_time) > announce_interval
                    )
                    if should_speak:
                        audio_bytes = speak(result, voice_rate)
                        if audio_bytes:
                            audio_placeholder.audio(audio_bytes, format="audio/mp3", autoplay=True)
                        st.session_state.last_spoken_text = result
                        st.session_state.last_spoken_time = now
            else:
                break

            if not webrtc_ctx.state.playing:
                break

with tab2:
    uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read this image file: {e}")
            image = None

        if image is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original Image", use_column_width=True)

            annotated, detections = run_detection(image, confidence_threshold)

            with col2:
                st.image(annotated, channels="BGR", caption="Detected Objects", use_column_width=True)

            if detections:
                scene_desc = describe_scene(detections)
                st.success(f"✅ {scene_desc}")
                audio_bytes = speak(scene_desc, voice_rate)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            else:
                st.warning("No objects detected with current confidence threshold.")

st.markdown("---")
st.caption("Powered by YOLOv8 | Built for Accessibility")