import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# Graceful import of ultralytics YOLO
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning(
        "⚠️ La bibliothèque `ultralytics` n'est pas installée. Mode démonstration activé."
    )

st.set_page_config(
    page_title="Road Accident AI System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css():
    """
    Inject custom CSS styles into the Streamlit app for a dark, modern theme.
    Defines accent colors, card styles, metric containers, and button overrides.
    """
    st.markdown(
        """
    <style>
        /* ── Global background ── */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }

        /* ── Sidebar styling ── */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #FF4B4B;
        }

        /* ── Main header ── */
        .main-header {
            background: linear-gradient(135deg, #1A1F2E 0%, #161B22 50%, #1A1F2E 100%);
            border: 1px solid #FF4B4B33;
            border-radius: 16px;
            padding: 28px 36px;
            margin-bottom: 28px;
            text-align: center;
        }
        .main-header h1 {
            color: #FF4B4B;
            font-size: 2.6rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            text-shadow: 0 0 20px #FF4B4B55;
        }
        .main-header p {
            color: #8B949E;
            font-size: 1.1rem;
            margin: 0;
        }

        /* ── Section title ── */
        .section-title {
            color: #FF4B4B;
            font-size: 1.6rem;
            font-weight: 700;
            border-left: 4px solid #FF4B4B;
            padding-left: 14px;
            margin: 20px 0 16px 0;
        }

        /* ── Metric cards ── */
        .metric-card {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 20px 16px;
            text-align: center;
            transition: border-color 0.3s ease;
        }
        .metric-card:hover {
            border-color: #FF4B4B88;
        }
        .metric-card .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #FF4B4B;
        }
        .metric-card .metric-label {
            font-size: 0.85rem;
            color: #8B949E;
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ── Safe metric (green) ── */
        .metric-card.safe .metric-value {
            color: #00C853;
        }

        /* ── Frame thumbnail card ── */
        .frame-card {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 12px;
            text-align: center;
        }
        .frame-card .frame-label {
            font-size: 0.8rem;
            color: #8B949E;
            margin-top: 6px;
        }
        .frame-card .frame-accident {
            color: #FF4B4B;
            font-weight: 600;
        }
        .frame-card .frame-safe {
            color: #00C853;
            font-weight: 600;
        }

        /* ── Info banner ── */
        .info-banner {
            background: linear-gradient(90deg, #1A2744 0%, #161B22 100%);
            border: 1px solid #3D5A99;
            border-radius: 10px;
            padding: 16px 20px;
            color: #90CAF9;
            margin: 14px 0;
        }

        /* ── Placeholder card ── */
        .placeholder-card {
            background-color: #161B22;
            border: 2px dashed #30363D;
            border-radius: 14px;
            padding: 40px 24px;
            text-align: center;
            color: #484F58;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .placeholder-card .ph-icon {
            font-size: 3rem;
            margin-bottom: 12px;
        }
        .placeholder-card .ph-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #6E7681;
            margin-bottom: 8px;
        }
        .placeholder-card .ph-desc {
            font-size: 0.85rem;
            color: #484F58;
        }

        /* ── Summary box ── */
        .summary-box {
            background: linear-gradient(135deg, #1A2744 0%, #1A1F2E 100%);
            border: 1px solid #FF4B4B44;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }
        .summary-box h3 {
            color: #FF4B4B;
            margin-top: 0;
        }

        /* ── Streamlit overrides ── */
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: background-color 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #CC3333;
        }
        .stButton > button:disabled {
            background-color: #30363D;
            color: #484F58;
        }
        div[data-testid="stMetric"] {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 10px;
            padding: 14px 16px;
        }
        div[data-testid="stMetric"] label {
            color: #8B949E !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FF4B4B;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #161B22;
            border-radius: 10px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #8B949E;
            border-radius: 8px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF4B4B22 !important;
            color: #FF4B4B !important;
        }
        .stProgress > div > div {
            background-color: #FF4B4B;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


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
    """
    Run accident detection on a single image frame.

    If a real YOLO model is provided, it runs inference and uses
    results[0].plot() to overlay bounding boxes. Otherwise, it falls back
    to the static rectangle demo mode.

    Args:
        image_np (np.ndarray): Input image as a NumPy array (RGB format).
        model: A loaded YOLO model instance, or None for fallback mode.

    Returns:
        tuple:
            - result_image (np.ndarray): Annotated image as NumPy array (RGB).
            - detections (list): List of dicts with keys 'confidence' and 'class'.
            - processing_time_ms (float): Inference time in milliseconds.
    """
    start_time = time.time()
    detections = []
    result_image = image_np.copy()

    try:
        if model is not None and YOLO_AVAILABLE:
            # ── Real YOLO inference path ──────────────────────────────────
            results = model(image_np, verbose=False)

            # results[0].plot() returns a BGR numpy array with boxes drawn
            plotted = results[0].plot()
            result_image = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

            # Extract detection metadata
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names.get(cls, f"Classe {cls}")
                    detections.append({"confidence": conf, "class": label})

        else:
            # ── Static fallback demo path ─────────────────────────────────
            # Convert to BGR for OpenCV drawing, then back to RGB
            bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            bgr_drawn = draw_static_detection(bgr)
            result_image = cv2.cvtColor(bgr_drawn, cv2.COLOR_BGR2RGB)
            detections = [{"confidence": 0.87, "class": "Accident (Demo)"}]

    except MemoryError:
        st.error("❌ Mémoire insuffisante pour traiter cette image.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la détection : {e}")

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


def display_detection_metrics(detections: list, processing_time: float):
    """
    Render a styled 3-column metrics row below a detection result image.

    Displays:
      - Number of accidents detected
      - Average confidence score (formatted as percentage)
      - Processing time in milliseconds

    Args:
        detections (list): List of detection dicts with 'confidence' key.
        processing_time (float): Time taken for inference in milliseconds.
    """
    num_detections = len(detections)
    avg_confidence = (
        sum(d["confidence"] for d in detections) / num_detections
        if num_detections > 0
        else 0.0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card {"safe" if num_detections == 0 else ""}">
            <div class="metric-value">{num_detections}</div>
            <div class="metric-label">🚨 Accidents Détectés</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{avg_confidence:.0%}</div>
            <div class="metric-label">📏 Confiance Moyenne</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card safe">
            <div class="metric-value">{processing_time:.0f} ms</div>
            <div class="metric-label">⏱️ Temps de Traitement</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Small spacer after the metric row
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)


# =============================================================================
# SECTION RENDERERS
# =============================================================================


def render_detection_section():
    """
    Render the complete 'Détection d'Accidents' section of the application.

    This includes:
      - Model loading via sidebar file uploader
      - Two tabs: Image Analysis and Video Analysis
      - Static fallback mode when no YOLO model is available
    """
    st.markdown(
        '<div class="section-title">🔍 Détection d\'Accidents</div>',
        unsafe_allow_html=True,
    )

    # ── MODEL LOADING VIA SIDEBAR ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 Modèle YOLOv8")

        model_file = st.file_uploader(
            "Charger votre modèle (.pt)",
            type=["pt"],
            help="Glissez-déposez votre modèle YOLOv8 entraîné au format .pt",
        )

        if model_file is not None:
            with st.spinner("Chargement du modèle..."):
                model = load_model(model_file.read())

            if model is not None:
                st.session_state["model"] = model
                st.success("✅ Modèle chargé avec succès !")
                st.caption(f"Fichier : `{model_file.name}`")
            else:
                st.error("❌ Échec du chargement.")
                st.session_state["model"] = None
        else:
            # No model file uploaded — keep whatever is in session state
            if "model" not in st.session_state:
                st.session_state["model"] = None

        # Display current model status
        if st.session_state.get("model") is not None:
            st.markdown(
                '<span style="color:#00C853;">● Modèle actif</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#FF4B4B;">● Mode démonstration</span>',
                unsafe_allow_html=True,
            )

    # ── NO MODEL WARNING ──────────────────────────────────────────────────────
    if st.session_state.get("model") is None:
        st.warning(
            "⚠️ Veuillez charger votre modèle YOLOv8 (.pt) dans la barre latérale. "
            "En attendant, le **mode démonstration** est actif avec une détection statique."
        )

    # ── TAB LAYOUT ────────────────────────────────────────────────────────────
    tab_image, tab_video = st.tabs(["🖼️ Analyse d'Image", "🎬 Analyse de Vidéo"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 : IMAGE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_image:
        st.markdown("#### 📂 Charger une image à analyser")

        uploaded_image = st.file_uploader(
            "Formats acceptés : JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
        )

        if uploaded_image is not None:
            try:
                # Load image with PIL and convert to numpy array (RGB)
                pil_image = Image.open(uploaded_image).convert("RGB")
                image_np = np.array(pil_image)

                st.markdown("---")
                col_orig, col_result = st.columns(2, gap="large")

                # LEFT — Original image
                with col_orig:
                    st.markdown(
                        "**🖼️ Image Originale**", help="Image uploadée sans traitement"
                    )
                    st.image(image_np, use_container_width=True)

                # RIGHT — Detection result
                with col_result:
                    st.markdown("**🔍 Résultat de Détection**")
                    with st.spinner("Analyse en cours..."):
                        result_np, detections, proc_time = detect_on_image(
                            image_np, st.session_state.get("model")
                        )
                    st.image(result_np, use_container_width=True)

                # Metrics row below both images
                st.markdown("#### 📊 Métriques de Détection")
                display_detection_metrics(detections, proc_time)

                # List detected classes if any
                if detections:
                    st.markdown("#### 🏷️ Détections")
                    for i, det in enumerate(detections, 1):
                        st.markdown(
                            f"**Objet {i}** — Classe : `{det['class']}` "
                            f"| Confiance : `{det['confidence']:.2%}`"
                        )

            except Exception as e:
                st.error(f"❌ Format d'image non supporté ou fichier corrompu : {e}")

        else:
            # Placeholder before upload
            st.markdown(
                """
            <div class="placeholder-card">
                <div class="ph-icon">🖼️</div>
                <div class="ph-title">Aucune image chargée</div>
                <div class="ph-desc">
                    Uploadez une image (JPG / JPEG / PNG) pour lancer la détection.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 : VIDEO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_video:
        st.markdown("#### 📂 Charger une vidéo à analyser")

        uploaded_video = st.file_uploader(
            "Formats acceptés : MP4, AVI, MOV",
            type=["mp4", "avi", "mov"],
            key="video_uploader",
        )

        # Frame extraction interval slider
        frame_interval = st.slider(
            "Intervalle d'extraction (frames)",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Extraire 1 frame toutes les N frames. "
            "Valeur plus élevée = traitement plus rapide.",
        )

        if uploaded_video is not None:
            st.markdown("---")
            launch_btn = st.button(
                "🎬 Lancer l'Analyse Vidéo", use_container_width=True
            )

            if launch_btn:
                video_bytes = uploaded_video.read()

                # ── STEP 1 : FRAME EXTRACTION ─────────────────────────────
                st.markdown("#### ⚙️ Étape 1 — Extraction des frames")
                with st.spinner("Extraction en cours..."):
                    frames = extract_frames(video_bytes, frame_interval)

                if not frames:
                    st.error("❌ Aucune frame extraite. Vérifiez le fichier vidéo.")
                    return

                st.success(f"✅ **{len(frames)} frames** extraites avec succès !")

                # ── STEP 2 : DETECTION ON EACH FRAME ─────────────────────
                st.markdown("#### ⚙️ Étape 2 — Détection sur chaque frame")

                detect_progress = st.progress(0.0)
                detect_status = st.empty()

                processed_frames = []
                all_confidences = []
                best_confidence = 0.0
                best_frame_index = None
                frames_with_accident = 0

                for i, (frame_idx, frame_np) in enumerate(frames):
                    detect_status.text(
                        f"Analyse en cours... Frame {i + 1}/{len(frames)}"
                    )

                    result_np, detections, proc_time = detect_on_image(
                        frame_np, st.session_state.get("model")
                    )

                    has_accident = len(detections) > 0
                    if has_accident:
                        frames_with_accident += 1
                        for det in detections:
                            all_confidences.append(det["confidence"])
                            if det["confidence"] > best_confidence:
                                best_confidence = det["confidence"]
                                best_frame_index = frame_idx

                    processed_frames.append(
                        {
                            "frame_idx": frame_idx,
                            "image": result_np,
                            "detections": detections,
                            "has_accident": has_accident,
                            "proc_time": proc_time,
                        }
                    )

                    detect_progress.progress((i + 1) / len(frames))

                detect_status.empty()
                detect_progress.progress(1.0)

                # ── STEP 3 : RESULTS DISPLAY ──────────────────────────────
                st.markdown("#### 📊 Étape 3 — Résultats de l'analyse")

                # Summary card
                total_analyzed = len(processed_frames)
                detection_rate = (
                    frames_with_accident / total_analyzed * 100
                    if total_analyzed > 0
                    else 0
                )
                avg_conf_overall = (
                    sum(all_confidences) / len(all_confidences)
                    if all_confidences
                    else 0.0
                )

                st.markdown(
                    f"""
                <div class="summary-box">
                    <h3>🏆 Résumé de l'Analyse Vidéo</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)

                with sum_c1:
                    st.metric("🎬 Frames Analysées", total_analyzed)
                with sum_c2:
                    st.metric("🚨 Frames avec Accident", frames_with_accident)
                with sum_c3:
                    st.metric("📈 Taux de Détection", f"{detection_rate:.1f}%")
                with sum_c4:
                    st.metric(
                        "⭐ Meilleure Confiance",
                        f"{best_confidence:.0%}"
                        if best_frame_index is not None
                        else "—",
                        help=(
                            f"Frame #{best_frame_index}"
                            if best_frame_index is not None
                            else ""
                        ),
                    )

                if best_frame_index is not None:
                    st.info(
                        f"📍 Détection maximale à la frame **#{best_frame_index}** "
                        f"avec une confiance de **{best_confidence:.0%}**"
                    )

                # Frame grid — 3 per row
                st.markdown("#### 🗂️ Galerie des Frames Analysées")
                st.caption(
                    f"Affichage de {len(processed_frames)} frames "
                    f"({'Modèle actif' if st.session_state.get('model') else 'Mode démo'})"
                )

                cols_per_row = 3
                for row_start in range(0, len(processed_frames), cols_per_row):
                    row_frames = processed_frames[row_start : row_start + cols_per_row]
                    grid_cols = st.columns(cols_per_row, gap="small")

                    for col_obj, frame_data in zip(grid_cols, row_frames):
                        with col_obj:
                            st.image(frame_data["image"], use_container_width=True)
                            if frame_data["has_accident"]:
                                best_det_conf = max(
                                    d["confidence"] for d in frame_data["detections"]
                                )
                                st.markdown(
                                    f"<div style='text-align:center;font-size:0.78rem;'>"
                                    f"<b>Frame #{frame_data['frame_idx']}</b><br>"
                                    f"<span style='color:#FF4B4B;'>🚨 Accident — "
                                    f"{best_det_conf:.0%}</span></div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div style='text-align:center;font-size:0.78rem;'>"
                                    f"<b>Frame #{frame_data['frame_idx']}</b><br>"
                                    f"<span style='color:#00C853;'>✅ Aucun accident</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(
                                "<div style='margin-bottom:12px;'></div>",
                                unsafe_allow_html=True,
                            )

        else:
            # Placeholder before upload
            st.markdown(
                """
            <div class="placeholder-card">
                <div class="ph-icon">🎬</div>
                <div class="ph-title">Aucune vidéo chargée</div>
                <div class="ph-desc">
                    Uploadez une vidéo (MP4 / AVI / MOV) pour lancer l'analyse frame par frame.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_prediction_section():
    """
    Render the 'Prédiction d'Accidents' section placeholder.

    This section is planned for a future ML-based prediction module
    that will estimate accident probability from environmental,
    meteorological, and traffic parameters.
    """
    st.markdown(
        '<div class="section-title">📊 Module de Prédiction d\'Accidents</div>',
        unsafe_allow_html=True,
    )


def main():
    """
    Main function — entry point for the Streamlit application.

    Handles:
      - CSS injection
      - Page header rendering
      - Sidebar navigation setup with session_state
      - Routing to the correct section renderer
    """
    # Inject global styles
    inject_custom_css()

    # Initialize navigation state
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = "detection"

    # Initialize model state
    if "model" not in st.session_state:
        st.session_state["model"] = None

    # ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center; padding: 12px 0 20px 0;">
            <div style="font-size:2.2rem;">🚨</div>
            <div style="font-size:1rem; font-weight:700; color:#FF4B4B;">
                Road Accident AI
            </div>
            <div style="font-size:0.72rem; color:#484F58;">
                Système de Détection & Prédiction
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📁 Navigation")

        # Detection button
        det_style = (
            "background:#FF4B4B22; border-left:3px solid #FF4B4B;"
            if st.session_state["active_section"] == "detection"
            else ""
        )
        if st.button(
            "🔍 Détection d'Accidents", use_container_width=True, key="nav_detection"
        ):
            st.session_state["active_section"] = "detection"
            st.rerun()

        # Prediction button
        if st.button(
            "📊 Prédiction d'Accidents", use_container_width=True, key="nav_prediction"
        ):
            st.session_state["active_section"] = "prediction"
            st.rerun()

    # ── GLOBAL HEADER ─────────────────────────────────────────────────────────
    st.markdown(
        """
    <div class="main-header">
        <h1>🚨 Road Accident AI System</h1>
        <p>
            Système intelligent de détection et de prédiction d'accidents routiers
            alimenté par l'Intelligence Artificielle et la Vision par Ordinateur
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── BREADCRUMB / ACTIVE INDICATOR ─────────────────────────────────────────
    active = st.session_state["active_section"]
    section_label = (
        "🔍 Détection d'Accidents"
        if active == "detection"
        else "📊 Prédiction d'Accidents"
    )
    st.markdown(
        f"<p style='color:#484F58; font-size:0.82rem; margin-bottom:4px;'>"
        f"Accueil › <b style='color:#8B949E;'>{section_label}</b></p>",
        unsafe_allow_html=True,
    )

    # ── SECTION ROUTING ───────────────────────────────────────────────────────
    if active == "detection":
        render_detection_section()
    elif active == "prediction":
        render_prediction_section()


if __name__ == "__main__":
    main()
