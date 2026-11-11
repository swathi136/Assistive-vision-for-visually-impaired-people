"""
Vision Intelligence Dashboard
- Background ML processing (non-blocking)
- Live webcam / video demo / image upload
- Voice feedback (throttled)
- Frame skipping for smooth display
"""

import io
import base64
import queue
import threading
import time

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import easyocr
from gtts import gTTS

# ──────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# 1. LOAD MODELS (cached – runs once per session)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models – this takes ~30 s the first time…")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    yolo = YOLO("yolov8n.pt")
    ocr  = easyocr.Reader(["en"], gpu=(device == "cuda"))
    return device, processor, blip, yolo, ocr

DEVICE, PROCESSOR, BLIP_MODEL, YOLO_MODEL, OCR_READER = load_models()

# ──────────────────────────────────────────────────────────────
# 2. VOICE
# ──────────────────────────────────────────────────────────────
def speak(text: str):
    """Inject an autoplay audio element into the page."""
    if not text:
        return
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    st.markdown(
        f'<audio autoplay src="data:audio/mp3;base64,{b64}">',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# 3. ML ANALYSIS (runs in background thread)
# ──────────────────────────────────────────────────────────────
def analyse_frame(frame_bgr: np.ndarray) -> dict:
    """Run YOLO + BLIP on one BGR frame. Returns result dict."""
    t0 = time.time()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil   = Image.fromarray(frame_rgb)

    # YOLO
    y_res  = YOLO_MODEL(frame_bgr, conf=0.5, verbose=False)
    labels = [YOLO_MODEL.names[int(b.cls[0])] for r in y_res for b in r.boxes]

    # Draw bounding boxes
    annotated = y_res[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # BLIP
    inputs = PROCESSOR(img_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = BLIP_MODEL.generate(**inputs, max_new_tokens=25)
    caption = PROCESSOR.decode(out[0], skip_special_tokens=True)

    latency = time.time() - t0
    return {
        "caption":      caption,
        "labels":       list(set(labels)),
        "annotated":    annotated_rgb,
        "latency":      latency,
    }


def background_worker(
    frame_q: queue.Queue,
    result_q: queue.Queue,
    stop_event: threading.Event,
):
    """
    Continuously pulls the *latest* frame from frame_q,
    processes it, and pushes the result to result_q.
    Skips stale frames so the queue never backs up.
    """
    while not stop_event.is_set():
        # Drain queue, keep only the newest frame
        latest = None
        try:
            while True:
                latest = frame_q.get_nowait()
        except queue.Empty:
            pass

        if latest is None:
            time.sleep(0.02)  # nothing to do yet
            continue

        result = analyse_frame(latest)

        # Also drain result_q (we only care about latest result)
        try:
            while True:
                result_q.get_nowait()
        except queue.Empty:
            pass

        result_q.put(result)

# ──────────────────────────────────────────────────────────────
# 4. IMAGE UPLOAD ANALYSIS
# ──────────────────────────────────────────────────────────────
def analyse_image_upload(uploaded_file, enable_voice: bool, enable_ocr: bool):
    img_pil  = Image.open(uploaded_file).convert("RGB")
    img_arr  = np.array(img_pil)
    frame_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    with st.spinner("Analysing image…"):
        result = analyse_frame(frame_bgr)
        if enable_ocr:
            ocr_results = OCR_READER.readtext(img_arr)
            ocr_texts   = [r[1] for r in ocr_results if r[2] > 0.4]
        else:
            ocr_texts = []

    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.image(result["annotated"], caption="Detected objects", use_container_width=True)
    with col_info:
        st.metric("Latency", f"{result['latency']:.2f}s")
        st.info(f"**📝 Description**\n\n{result['caption']}")
        if result["labels"]:
            st.success(f"**🔍 Objects**\n\n{', '.join(result['labels'])}")
        else:
            st.warning("No objects detected above threshold.")
        if ocr_texts:
            st.info(f"**🔤 OCR Text**\n\n{chr(10).join(ocr_texts)}")
        if enable_voice:
            speak(result["caption"])

# ──────────────────────────────────────────────────────────────
# 5. VIDEO / WEBCAM LOOP
# ──────────────────────────────────────────────────────────────
def run_video_loop(
    cap: cv2.VideoCapture,
    enable_voice: bool,
    display_annotated: bool,
    viewfinder,
    cap_ph,
    obj_ph,
    perf_ph,
):
    frame_q: queue.Queue = queue.Queue(maxsize=2)
    result_q: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    worker = threading.Thread(
        target=background_worker,
        args=(frame_q, result_q, stop_event),
        daemon=True,
    )
    worker.start()

    last_result    = None
    last_voice_t   = 0
    frame_idx      = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Push to ML queue every 3rd frame (skip the rest)
            if frame_idx % 3 == 0:
                try:
                    frame_q.put_nowait(frame.copy())
                except queue.Full:
                    pass  # drop – worker is busy with a newer frame
            frame_idx += 1

            # Non-blocking pull of latest ML result
            try:
                last_result = result_q.get_nowait()
            except queue.Empty:
                pass

            # ── Display ──────────────────────────────────────
            if display_annotated and last_result is not None:
                viewfinder.image(
                    last_result["annotated"],
                    channels="RGB",
                    use_container_width=True,
                )
            else:
                viewfinder.image(frame_rgb, channels="RGB", use_container_width=True)

            if last_result:
                perf_ph.metric("Latency", f"{last_result['latency']:.2f}s")
                cap_ph.info(f"**📝 Description**\n\n{last_result['caption']}")
                obj_ph.success(
                    f"**🔍 Objects**\n\n"
                    + (", ".join(last_result["labels"]) if last_result["labels"] else "None")
                )
                if enable_voice and (time.time() - last_voice_t > 5):
                    speak(last_result["caption"])
                    last_voice_t = time.time()

            # Small sleep keeps Streamlit responsive; tune as needed
            time.sleep(0.03)

    finally:
        stop_event.set()
        worker.join(timeout=3)
        cap.release()

# ──────────────────────────────────────────────────────────────
# 6. SIDEBAR
# ──────────────────────────────────────────────────────────────
st.sidebar.title("🎮 Control Panel")

mode = st.sidebar.radio(
    "Input Source",
    ["📷 Live Webcam", "🎬 Video Demo", "🖼️ Image Upload"],
    index=0,
)

video_file   = None
upload_file  = None

if mode == "🎬 Video Demo":
    video_file = st.sidebar.file_uploader(
        "Upload Demo Video", type=["mp4", "mov", "avi"]
    )
elif mode == "🖼️ Image Upload":
    upload_file = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

st.sidebar.divider()
enable_voice     = st.sidebar.toggle("🔊 Voice Feedback",       value=True)
display_annotated = st.sidebar.toggle("🟩 Show Bounding Boxes", value=True)
enable_ocr       = st.sidebar.toggle("🔤 OCR (Image Upload)",   value=False)

st.sidebar.divider()
st.sidebar.caption(
    "⚡ ML runs in a background thread.\n"
    "Frames are displayed at ~30 fps;\n"
    "AI results update as they arrive."
)

# ──────────────────────────────────────────────────────────────
# 7. MAIN UI
# ──────────────────────────────────────────────────────────────
st.title("👁️ Vision Intelligence Dashboard")

if mode == "🖼️ Image Upload":
    st.subheader("Image Analysis")
    if upload_file:
        analyse_image_upload(upload_file, enable_voice, enable_ocr)
    else:
        st.info("Upload an image from the sidebar to begin analysis.")

else:
    c1, c2 = st.columns([2, 1])
    with c1:
        viewfinder = st.empty()
        run_btn = st.button(
            "▶ Start Processing", type="primary", use_container_width=True
        )
    with c2:
        st.subheader("Real-Time Analysis")
        cap_ph   = st.empty()
        obj_ph   = st.empty()
        perf_ph  = st.empty()

    if run_btn:
        if mode == "🎬 Video Demo":
            if video_file is None:
                st.error("Please upload a video file first.")
                st.stop()
            vid_bytes = video_file.read()
            with open("_temp_vid.mp4", "wb") as f:
                f.write(vid_bytes)
            cap = cv2.VideoCapture("_temp_vid.mp4")
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open video source. Check webcam or video file.")
            st.stop()

        run_video_loop(
            cap,
            enable_voice,
            display_annotated,
            viewfinder,
            cap_ph,
            obj_ph,
            perf_ph,
        )
        st.success("✅ Processing complete.")