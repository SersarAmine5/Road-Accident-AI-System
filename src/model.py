import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

# Graceful import of ultralytics YOLO
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def load_model(model_bytes: bytes):
    """
    Load a YOLOv8 model from raw bytes using a temporary file.

    This function is cached with st.cache_resource so the model is
    loaded only once per session, even across Streamlit reruns.

    Args:
        model_bytes (bytes): Raw bytes of the .pt model file uploaded by the user.

    Returns:
        YOLO | None: A loaded YOLO model instance, or None if loading fails
                     or if ultralytics is not available.
    """
    if not YOLO_AVAILABLE:
        return None

    try:
        # Write bytes to a temporary .pt file so YOLO can load it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(model_bytes)
            tmp_path = tmp_file.name

        model = YOLO(tmp_path)
        os.unlink(tmp_path)  # Clean up temp file after loading
        return model

    except Exception as e:
        st.error(f"❌ Échec du chargement du modèle : {e}")
        return None


def draw_static_detection(image_np: np.ndarray) -> np.ndarray:
    """
    Draw a static demonstration bounding box on an image when no YOLO model
    is available. Used as a mandatory fallback for demo purposes.

    Draws:
      - A red rectangle covering ~30% of width & height, centered in the image
      - A red label "Accident Détecté (Demo)" above the rectangle
      - A yellow confidence score "Conf: 0.87" inside the rectangle

    Args:
        image_np (np.ndarray): Input image as a NumPy array (BGR or RGB).

    Returns:
        np.ndarray: A copy of the image with the static annotation drawn on it.
    """
    result = image_np.copy()
    h, w = result.shape[:2]

    # Calculate bounding box coordinates (30% of image, centered)
    box_w = int(w * 0.30)
    box_h = int(h * 0.30)
    cx, cy = w // 2, h // 2
    x1 = cx - box_w // 2
    y1 = cy - box_h // 2
    x2 = cx + box_w // 2
    y2 = cy + box_h // 2

    # Colors
    RED = (0, 0, 255)  # BGR
    YELLOW = (0, 255, 255)  # BGR
    WHITE = (255, 255, 255)  # BGR
    BG_RED = (0, 0, 180)

    # Draw filled label background above box
    label_text = "Accident Detecte (Demo)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 800)
    thickness = max(1, int(min(w, h) / 400))

    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    label_y1 = max(y1 - th - 10, 0)
    label_y2 = y1

    cv2.rectangle(result, (x1, label_y1), (x1 + tw + 8, label_y2), BG_RED, -1)
    cv2.putText(
        result,
        label_text,
        (x1 + 4, label_y2 - 4),
        font,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )

    # Draw main bounding box (thick red border)
    border = max(2, thickness * 2)
    cv2.rectangle(result, (x1, y1), (x2, y2), RED, border)

    # Draw confidence score inside box
    conf_text = "Conf: 0.87"
    cv2.putText(
        result,
        conf_text,
        (x1 + 8, y2 - 10),
        font,
        font_scale,
        YELLOW,
        thickness,
        cv2.LINE_AA,
    )

    return result


def detect_on_image(image_np: np.ndarray, model) -> tuple[np.ndarray, list, float]:
    start_time = time.time()
    detections = []
    result_image = image_np.copy()

    try:
        if model is not None and YOLO_AVAILABLE:
            results = model.predict(image_np, verbose=False, conf=0.5) # use .predict() not __call__

            # results[0].plot() returns BGR — convert to RGB
            plotted = results[0].plot()
            result_image = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # model.names is a dict like {0: 'accident', 1: 'vehicle'}
                names = (
                    model.names
                    if isinstance(model.names, dict)
                    else {i: n for i, n in enumerate(model.names)}
                )
                for box in boxes:
                 conf = float(box.conf[0])
                 cls = int(box.cls[0])
                 label = names.get(cls, f"Classe {cls}")
                 if label.lower() == "accident" and conf >= 0.50:
                     detections.append({"confidence": conf, "class": label})
        else:
            bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            bgr_drawn = draw_static_detection(bgr)
            result_image = cv2.cvtColor(bgr_drawn, cv2.COLOR_BGR2RGB)
            detections = [{"confidence": 0.87, "class": "Accident (Demo)"}]

    except MemoryError:
        st.error("❌ Mémoire insuffisante pour traiter cette image.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la détection : {e}")
        st.exception(e)  # ← shows full traceback in the UI during debugging

    processing_time_ms = (time.time() - start_time) * 1000
    return result_image, detections, processing_time_ms


@st.cache_data(show_spinner=False)
def extract_frames(video_bytes: bytes, frame_interval: int) -> list:
    """
    Extract frames from a video at a given frame interval.

    Frames are extracted as RGB NumPy arrays and capped at 100 frames
    to prevent memory overflow. Uses a temporary file to handle the
    uploaded video bytes.

    Args:
        video_bytes (bytes): Raw bytes of the uploaded video file.
        frame_interval (int): Extract one frame every N frames.

    Returns:
        list: A list of (frame_index, np.ndarray) tuples in RGB format.
              Returns an empty list if extraction fails.
    """
    frames = []

    try:
        # Write video bytes to a temp file for cv2.VideoCapture
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            st.error("❌ Impossible d'ouvrir le fichier vidéo. Vérifiez le format.")
            os.unlink(tmp_path)
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        MAX_FRAMES = 100  # Hard cap to avoid memory overflow

        # Create progress UI elements
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR (OpenCV) → RGB (PIL / Streamlit)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_count, rgb_frame))

                # Update progress bar
                progress = min(frame_count / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)
                status_text.text(
                    f"Extraction des frames... {frame_count}/{total_frames}"
                )

                if len(frames) >= MAX_FRAMES:
                    status_text.text(
                        f"⚠️ Limite de {MAX_FRAMES} frames atteinte. "
                        "Arrêt de l'extraction."
                    )
                    break

            frame_count += 1

        cap.release()
        os.unlink(tmp_path)

        progress_bar.progress(1.0)
        status_text.empty()

    except MemoryError:
        st.error("❌ Mémoire insuffisante pour traiter cette vidéo.")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction des frames : {e}")

    return frames
