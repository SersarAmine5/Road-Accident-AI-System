import os
import tempfile
import time
import pandas as pd
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from model import YOLO_AVAILABLE, detect_on_image, extract_frames, load_model

if not YOLO_AVAILABLE:
    st.warning("⚠️ La bibliothèque `ultralytics` n'est pas installée. Mode démonstration activé.")

st.set_page_config(
    page_title="Road Accident AI System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css():
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

        /* ── GLOBAL ── */
        .stApp {
            background-color: #070B14;
            color: #CBD5E1;
            font-family: 'Inter', sans-serif;
        }

        /* ── SIDEBAR ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D1423 0%, #070B14 100%);
            border-right: 1px solid rgba(0, 255, 255, 0.08);
        }

        /* ── MAIN HEADER ── */
        .cyber-header {
            position: relative;
            background: linear-gradient(135deg, #0D1423 0%, #0A1628 50%, #0D1423 100%);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 4px;
            padding: 36px 48px;
            margin-bottom: 28px;
            text-align: center;
            overflow: hidden;
        }
        .cyber-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00D4FF, #0066FF, #00D4FF, transparent);
        }
        .cyber-header::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
        }
        .cyber-header h1 {
            font-family: 'Rajdhani', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin: 0 0 8px 0;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
        }
        .cyber-header .accent {
            color: #00D4FF;
        }
        .cyber-header p {
            font-family: 'Share Tech Mono', monospace;
            color: rgba(0, 212, 255, 0.6);
            font-size: 0.85rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin: 0;
        }

        /* ── SECTION TITLE ── */
        .section-title {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #00D4FF;
            letter-spacing: 2px;
            text-transform: uppercase;
            border-left: 3px solid #00D4FF;
            padding-left: 14px;
            margin: 24px 0 20px 0;
        }

        /* ── NAV LOGO ── */
        .nav-logo {
            text-align: center;
            padding: 20px 0 28px 0;
            border-bottom: 1px solid rgba(0,212,255,0.1);
            margin-bottom: 20px;
        }
        .nav-logo .logo-icon {
            font-size: 2.4rem;
            display: block;
            margin-bottom: 8px;
            filter: drop-shadow(0 0 8px rgba(0,212,255,0.5));
        }
        .nav-logo .logo-title {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .nav-logo .logo-sub {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.65rem;
            color: rgba(0,212,255,0.5);
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 2px;
        }

        /* ── NAV LABEL ── */
        .nav-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.65rem;
            color: rgba(0,212,255,0.4);
            letter-spacing: 3px;
            text-transform: uppercase;
            padding: 0 4px;
            margin-bottom: 8px;
        }

        /* ── BUTTONS ── */
        .stButton > button {
            background: transparent;
            color: #94A3B8;
            border: 1px solid rgba(0,212,255,0.15);
            border-radius: 2px;
            padding: 10px 20px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton > button:hover {
            background: rgba(0,212,255,0.08);
            color: #00D4FF;
            border-color: rgba(0,212,255,0.4);
            box-shadow: 0 0 12px rgba(0,212,255,0.1);
        }
        .stButton > button:active,
        .stButton > button:focus {
            background: rgba(0,212,255,0.12);
            color: #00D4FF;
            border-color: #00D4FF;
        }

        /* ── METRIC CARDS ── */
        .cyber-metric {
            background: linear-gradient(135deg, #0D1423, #0A1628);
            border: 1px solid rgba(0,212,255,0.12);
            border-radius: 3px;
            padding: 18px 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .cyber-metric::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
        }
        .cyber-metric .metric-value {
            font-family: 'Share Tech Mono', monospace;
            font-size: 1.8rem;
            color: #00D4FF;
            display: block;
        }
        .cyber-metric .metric-value.danger { color: #FF4B4B; }
        .cyber-metric .metric-value.safe { color: #00FF88; }
        .cyber-metric .metric-value.warn { color: #FFB800; }
        .cyber-metric .metric-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.62rem;
            color: rgba(0,212,255,0.45);
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 4px;
        }

        /* ── INFO BANNER ── */
        .cyber-banner {
            background: rgba(0,212,255,0.04);
            border: 1px solid rgba(0,212,255,0.15);
            border-left: 3px solid #00D4FF;
            border-radius: 2px;
            padding: 12px 18px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.8rem;
            color: rgba(0,212,255,0.7);
            letter-spacing: 0.5px;
            margin: 14px 0;
        }

        /* ── SUMMARY BOX ── */
        .cyber-summary {
            background: linear-gradient(135deg, rgba(0,212,255,0.04), rgba(0,102,255,0.04));
            border: 1px solid rgba(0,212,255,0.15);
            border-radius: 3px;
            padding: 18px 22px;
            margin-bottom: 20px;
        }
        .cyber-summary h3 {
            font-family: 'Rajdhani', sans-serif;
            color: #00D4FF;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 0;
            font-size: 1rem;
        }

        /* ── PLACEHOLDER ── */
        .cyber-placeholder {
            background: rgba(0,212,255,0.02);
            border: 1px dashed rgba(0,212,255,0.12);
            border-radius: 3px;
            padding: 48px 24px;
            text-align: center;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .cyber-placeholder .ph-icon { font-size: 2.5rem; margin-bottom: 14px; opacity: 0.5; }
        .cyber-placeholder .ph-title {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            color: rgba(0,212,255,0.4);
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .cyber-placeholder .ph-desc {
            font-size: 0.8rem;
            color: rgba(148,163,184,0.4);
        }

        /* ── BREADCRUMB ── */
        .cyber-breadcrumb {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.72rem;
            color: rgba(0,212,255,0.35);
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .cyber-breadcrumb span { color: rgba(0,212,255,0.6); }

        /* ── STREAMLIT OVERRIDES ── */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #0D1423, #0A1628);
            border: 1px solid rgba(0,212,255,0.12);
            border-radius: 3px;
            padding: 14px 16px;
        }
        div[data-testid="stMetric"] label {
            font-family: 'Share Tech Mono', monospace !important;
            font-size: 0.65rem !important;
            color: rgba(0,212,255,0.45) !important;
            letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: 'Share Tech Mono', monospace;
            color: #00D4FF;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(0,212,255,0.03);
            border-bottom: 1px solid rgba(0,212,255,0.1);
            border-radius: 0;
            padding: 0;
            gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: rgba(148,163,184,0.6);
            border-bottom: 2px solid transparent;
            border-radius: 0;
            padding: 12px 20px;
        }
        .stTabs [aria-selected="true"] {
            background: transparent !important;
            color: #00D4FF !important;
            border-bottom: 2px solid #00D4FF !important;
        }
        .stProgress > div > div { background-color: #00D4FF; }
        .stSlider [data-baseweb="slider"] { }
        div[data-testid="stFileUploader"] {
            background: rgba(0,212,255,0.02);
            border: 1px dashed rgba(0,212,255,0.15);
            border-radius: 3px;
        }
        .stSelectbox [data-baseweb="select"] > div {
            background: #0D1423;
            border-color: rgba(0,212,255,0.2);
        }
        .stCheckbox label { color: #94A3B8; }
        h4 {
            font-family: 'Rajdhani', sans-serif;
            color: #CBD5E1;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .stSuccess {
            background: rgba(0,255,136,0.08) !important;
            border-color: rgba(0,255,136,0.3) !important;
        }
        .stWarning {
            background: rgba(255,184,0,0.08) !important;
            border-color: rgba(255,184,0,0.3) !important;
        }
        .stError {
            background: rgba(255,75,75,0.08) !important;
            border-color: rgba(255,75,75,0.3) !important;
        }
        /* scrollbar */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #070B14; }
        ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 2px; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def display_detection_metrics(detections: list, processing_time: float):
    num_detections = len(detections)
    avg_confidence = (
        sum(d["confidence"] for d in detections) / num_detections
        if num_detections > 0 else 0.0
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        val_class = "safe" if num_detections == 0 else "danger"
        st.markdown(
            f'<div class="cyber-metric"><span class="metric-value {val_class}">{num_detections}</span>'
            f'<div class="metric-label">// Accidents Détectés</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="cyber-metric"><span class="metric-value">{avg_confidence:.0%}</span>'
            f'<div class="metric-label">// Confiance Moyenne</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="cyber-metric"><span class="metric-value safe">{processing_time:.0f}ms</span>'
            f'<div class="metric-label">// Temps de Traitement</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)


def render_detection_section():
    st.markdown('<div class="section-title">// Détection d\'Accidents</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div style="border-top:1px solid rgba(0,212,255,0.1);margin:16px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">// Modèle</div>', unsafe_allow_html=True)
        model_file = st.file_uploader("Charger votre modèle (.pt)", type=["pt"])
        if model_file is not None:
            with st.spinner("Chargement..."):
                model = load_model(model_file.read())
            if model is not None:
                st.session_state["model"] = model
                st.success("✅ Modèle chargé")
                st.caption(f"`{model_file.name}`")
            else:
                st.error("❌ Échec du chargement.")
                st.session_state["model"] = None
        else:
            if "model" not in st.session_state:
                st.session_state["model"] = None
        if st.session_state.get("model") is not None:
            st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;color:#00FF88;letter-spacing:1px;">● MODÈLE ACTIF</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;color:rgba(255,75,75,0.8);letter-spacing:1px;">● MODE DÉMONSTRATION</div>', unsafe_allow_html=True)

    if st.session_state.get("model") is None:
        st.warning("⚠️ Chargez votre modèle YOLOv8 (.pt) dans la barre latérale.")

    tab_image, tab_video = st.tabs(["  Image  ", "  Vidéo  "])

    with tab_image:
        st.markdown("#### 📂 Charger une image")
        uploaded_image = st.file_uploader("JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], key="image_uploader")
        if uploaded_image is not None:
            try:
                pil_image = Image.open(uploaded_image).convert("RGB")
                image_np = np.array(pil_image)
                st.markdown('<div style="border-top:1px solid rgba(0,212,255,0.1);margin:16px 0;"></div>', unsafe_allow_html=True)
                col_orig, col_result = st.columns(2, gap="large")
                with col_orig:
                    st.markdown("**// IMAGE ORIGINALE**")
                    st.image(image_np, use_container_width=True)
                with col_result:
                    st.markdown("**// RÉSULTAT**")
                    with st.spinner("Analyse..."):
                        result_np, detections, proc_time = detect_on_image(image_np, st.session_state.get("model"))
                    st.image(result_np, use_container_width=True)
                st.markdown("#### // Métriques")
                display_detection_metrics(detections, proc_time)
                if detections:
                    st.markdown("#### // Détections")
                    for i, det in enumerate(detections, 1):
                        st.markdown(f"`[{i:02d}]` Classe : **{det['class']}** — Confiance : `{det['confidence']:.2%}`")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
        else:
            st.markdown(
                '<div class="cyber-placeholder"><div class="ph-icon">🖼️</div>'
                '<div class="ph-title">Aucune image chargée</div>'
                '<div class="ph-desc">Uploadez une image JPG / JPEG / PNG</div></div>',
                unsafe_allow_html=True,
            )

    with tab_video:
        st.markdown("#### 📂 Charger une vidéo")
        uploaded_video = st.file_uploader("MP4, AVI, MOV", type=["mp4", "avi", "mov"], key="video_uploader")
        frame_interval = st.slider("Intervalle d'extraction (frames)", min_value=5, max_value=30, value=10, step=1)
        if uploaded_video is not None:
            st.markdown('<div style="border-top:1px solid rgba(0,212,255,0.1);margin:16px 0;"></div>', unsafe_allow_html=True)
            launch_btn = st.button("▶  LANCER L'ANALYSE VIDÉO", use_container_width=True)
            if launch_btn:
                video_bytes = uploaded_video.read()
                st.markdown("#### // Étape 1 — Extraction")
                with st.spinner("Extraction des frames..."):
                    frames = extract_frames(video_bytes, frame_interval)
                if not frames:
                    st.error("❌ Aucune frame extraite.")
                    return
                st.success(f"✅ {len(frames)} frames extraites")
                st.markdown("#### // Étape 2 — Détection")
                detect_progress = st.progress(0.0)
                detect_status = st.empty()
                processed_frames, all_confidences = [], []
                best_confidence, best_frame_index, frames_with_accident = 0.0, None, 0
                for i, (frame_idx, frame_np) in enumerate(frames):
                    detect_status.text(f"Frame {i + 1}/{len(frames)}")
                    result_np, detections, proc_time = detect_on_image(frame_np, st.session_state.get("model"))
                    has_accident = any(d['class'].lower() == 'accident' for d in detections)
                    if has_accident:
                        frames_with_accident += 1
                        for det in detections:
                            all_confidences.append(det["confidence"])
                            if det["confidence"] > best_confidence:
                                best_confidence = det["confidence"]
                                best_frame_index = frame_idx
                    processed_frames.append({"frame_idx": frame_idx, "image": result_np, "detections": detections, "has_accident": has_accident, "proc_time": proc_time})
                    detect_progress.progress((i + 1) / len(frames))
                detect_status.empty()
                detect_progress.progress(1.0)
                st.markdown("#### // Résultats")
                total_analyzed = len(processed_frames)
                detection_rate = frames_with_accident / total_analyzed * 100 if total_analyzed > 0 else 0
                st.markdown(
                    '<div class="cyber-summary"><h3>// Résumé Analyse Vidéo</h3></div>',
                    unsafe_allow_html=True,
                )
                sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
                with sum_c1: st.metric("Frames Analysées", total_analyzed)
                with sum_c2: st.metric("Frames Accident", frames_with_accident)
                with sum_c3: st.metric("Taux Détection", f"{detection_rate:.1f}%")
                with sum_c4: st.metric("Meilleure Conf.", f"{best_confidence:.0%}" if best_frame_index is not None else "—")
                if best_frame_index is not None:
                    st.info(f"📍 Frame **#{best_frame_index}** — confiance max **{best_confidence:.0%}**")
                st.markdown("#### // Frames avec Accidents")
                cols_per_row = 3
                accident_frames = [f for f in processed_frames if f['has_accident']]
                if not accident_frames:
                    st.warning("⚠️ Aucun accident détecté.")
                else:
                    st.caption(f"{len(accident_frames)} / {len(processed_frames)} frames")
                    for row_start in range(0, len(accident_frames), cols_per_row):
                        row_frames = accident_frames[row_start: row_start + cols_per_row]
                        grid_cols = st.columns(cols_per_row, gap="small")
                        for col_obj, frame_data in zip(grid_cols, row_frames):
                            with col_obj:
                                st.image(frame_data["image"], use_container_width=True)
                                best_det_conf = max(d["confidence"] for d in frame_data["detections"])
                                st.markdown(
                                    f"<div style='text-align:center;font-family:Share Tech Mono,monospace;font-size:0.72rem;'>"
                                    f"<span style='color:rgba(0,212,255,0.5);'>FRAME #{frame_data['frame_idx']}</span><br>"
                                    f"<span style='color:#FF4B4B;'>🚨 {best_det_conf:.0%}</span></div>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="cyber-placeholder"><div class="ph-icon">🎬</div>'
                '<div class="ph-title">Aucune vidéo chargée</div>'
                '<div class="ph-desc">Uploadez une vidéo MP4 / AVI / MOV</div></div>',
                unsafe_allow_html=True,
            )


def render_prediction_section():
    st.markdown('<div class="section-title">// Prédiction de Risque</div>', unsafe_allow_html=True)

    @st.cache_resource
    def load_prediction_model():
        import joblib
        return joblib.load("models/model_xgboost.pkl")

    try:
        pred_model = load_prediction_model()
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle : {e}")
        return

    st.markdown(
        '<div class="cyber-banner">[ INPUT ] — Renseignez les conditions pour estimer le risque d\'accident grave</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:rgba(0,212,255,0.5);letter-spacing:2px;margin-bottom:12px;">// TEMPOREL</div>', unsafe_allow_html=True)
        hour = st.slider("Heure", 0, 23, 8)
        day_of_week = st.selectbox("Jour", options=[0,1,2,3,4,5,6],
            format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][x])
        month = st.selectbox("Mois", options=list(range(1,13)),
            format_func=lambda x: ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"][x-1])
        is_weekend = 1 if day_of_week >= 5 else 0

    with col2:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:rgba(0,212,255,0.5);letter-spacing:2px;margin-bottom:12px;">// MÉTÉO</div>', unsafe_allow_html=True)
        temperature = st.slider("Température (°F)", -20, 120, 65)
        humidity = st.slider("Humidité (%)", 0, 100, 60)
        visibility = st.slider("Visibilité (mi)", 0.0, 10.0, 9.0, step=0.1)
        wind_speed = st.slider("Vent (mph)", 0.0, 60.0, 8.0, step=0.5)
        is_rain = st.checkbox("🌧️ Pluie")
        is_snow = st.checkbox("❄️ Neige")
        is_fog = st.checkbox("🌫️ Brouillard")

    with col3:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:rgba(0,212,255,0.5);letter-spacing:2px;margin-bottom:12px;">// ROUTE</div>', unsafe_allow_html=True)
        sunrise_sunset = st.selectbox("Luminosité", ["Jour", "Nuit"])
        junction = st.checkbox("🔀 Intersection")
        traffic_signal = st.checkbox("🚦 Feu")
        crossing = st.checkbox("🚶 Passage piéton")
        give_way = st.checkbox("⚠️ Cédez le passage")
        station = st.checkbox("🚌 Station")
        stop = st.checkbox("🛑 Stop")

    st.markdown('<div style="border-top:1px solid rgba(0,212,255,0.1);margin:20px 0;"></div>', unsafe_allow_html=True)
    predict_btn = st.button("▶  ANALYSER LE RISQUE", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'hour': hour, 'day_of_week': day_of_week, 'month': month, 'is_weekend': is_weekend,
            'Temperature(F)': temperature, 'Humidity(%)': humidity,
            'Visibility(mi)': visibility, 'Wind_Speed(mph)': wind_speed,
            'Junction': int(junction), 'Traffic_Signal': int(traffic_signal),
            'Crossing': int(crossing), 'Sunrise_Sunset': 1 if sunrise_sunset == "Nuit" else 0,
            'Give_Way': int(give_way), 'Station': int(station), 'Stop': int(stop),
            'is_rain': int(is_rain), 'is_snow': int(is_snow), 'is_fog': int(is_fog),
        }])
        proba = pred_model.predict_proba(input_data)[0][1]
        pct = proba * 100

        if pct >= 40:
            color, glow, status, bar_color = "#FF4B4B", "rgba(255,75,75,0.3)", "RISQUE ÉLEVÉ", "#FF4B4B"
        elif pct >= 20:
            color, glow, status, bar_color = "#FFB800", "rgba(255,184,0,0.3)", "RISQUE MODÉRÉ", "#FFB800"
        else:
            color, glow, status, bar_color = "#00FF88", "rgba(0,255,136,0.3)", "RISQUE FAIBLE", "#00FF88"

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #0D1423, #0A1628);
                border: 1px solid {color}44;
                border-top: 2px solid {color};
                border-radius: 3px;
                padding: 40px 32px;
                text-align: center;
                margin-top: 20px;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 0.7rem;
                    color: rgba(0,212,255,0.4);
                    letter-spacing: 3px;
                    text-transform: uppercase;
                    margin-bottom: 16px;
                ">[ ANALYSE TERMINÉE ]</div>
                <div style="
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 4.5rem;
                    font-weight: 700;
                    color: {color};
                    text-shadow: 0 0 30px {glow};
                    line-height: 1;
                    margin-bottom: 8px;
                ">{pct:.1f}%</div>
                <div style="
                    font-family: 'Rajdhani', sans-serif;
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: {color};
                    letter-spacing: 4px;
                    text-transform: uppercase;
                    margin-bottom: 20px;
                ">{status}</div>
                <div style="
                    background: rgba(0,0,0,0.3);
                    border-radius: 2px;
                    height: 4px;
                    width: 60%;
                    margin: 0 auto 16px;
                    overflow: hidden;
                ">
                    <div style="
                        background: {bar_color};
                        height: 100%;
                        width: {min(pct, 100):.0f}%;
                        box-shadow: 0 0 8px {glow};
                    "></div>
                </div>
                <div style="
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 0.72rem;
                    color: rgba(148,163,184,0.5);
                    letter-spacing: 1px;
                ">Probabilité d'accident grave — XGBoost v1.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_map_section():
    st.markdown('<div class="section-title">// Carte de Risque</div>', unsafe_allow_html=True)

    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    import joblib

    @st.cache_resource
    def load_map_model():
        return joblib.load("models/model_xgboost.pkl")

    @st.cache_data
    def load_grid_data():
        df = pd.read_csv("data/US_Accidents_March23.csv",nrows=500000)
        df = df.sample(n=500000, random_state=42).reset_index(drop=True)
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed', errors='coerce')
        df['hour'] = df['Start_Time'].dt.hour
        df['day_of_week'] = df['Start_Time'].dt.dayofweek
        df['month'] = df['Start_Time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['lat_grid'] = df['Start_Lat'].round(1)
        df['lng_grid'] = df['Start_Lng'].round(1)
        df['is_rain'] = df['Weather_Condition'].str.contains('Rain', na=False).astype(int)
        df['is_snow'] = df['Weather_Condition'].str.contains('Snow', na=False).astype(int)
        df['is_fog'] = df['Weather_Condition'].str.contains('Fog|Haze', na=False).astype(int)
        df['Sunrise_Sunset_enc'] = (df['Sunrise_Sunset'] == 'Night').astype(int)
        return df

    @st.cache_data
    def compute_risk_grid(_model):
        df = load_grid_data()
        features = ['lat_grid', 'lng_grid', 'hour', 'day_of_week', 'month', 'is_weekend',
                    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)',
                    'Junction', 'Traffic_Signal', 'Crossing', 'Sunrise_Sunset_enc',
                    'Give_Way', 'Station', 'Stop', 'is_rain', 'is_snow', 'is_fog']
        grid = df[features].dropna()
        grid_agg = grid.groupby(['lat_grid', 'lng_grid']).mean().reset_index()
        X = grid_agg.drop(['lat_grid', 'lng_grid'], axis=1)
        X.columns = ['hour', 'day_of_week', 'month', 'is_weekend',
                     'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)',
                     'Junction', 'Traffic_Signal', 'Crossing', 'Sunrise_Sunset',
                     'Give_Way', 'Station', 'Stop', 'is_rain', 'is_snow', 'is_fog']
        grid_agg['risk_score'] = _model.predict_proba(X)[:, 1]
        return grid_agg

    st.markdown(
        '<div class="cyber-banner">[ DATA ] — Densité de risque basée sur 1M d\'accidents réels — US Accidents Database 2016–2023</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Calcul du risque géographique..."):
        map_model = load_map_model()
        grid_agg = compute_risk_grid(map_model)

    m = folium.Map(location=[37.5, -96.0], zoom_start=5, tiles='CartoDB dark_matter')
    heat_data = grid_agg[['lat_grid', 'lng_grid', 'risk_score']].values.tolist()
    HeatMap(heat_data, min_opacity=0.3, radius=15, blur=20).add_to(m)
    st_folium(m, width=None, height=560)


def main():
    inject_custom_css()

    if "active_section" not in st.session_state:
        st.session_state["active_section"] = "detection"
    if "model" not in st.session_state:
        st.session_state["model"] = None

    with st.sidebar:
        st.markdown(
            """
            <div class="nav-logo">
                <span class="logo-icon">🚨</span>
                <div class="logo-title">Road Accident AI</div>
                <div class="logo-sub">Detection & Prediction System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-label">// Navigation</div>', unsafe_allow_html=True)

        if st.button("  Détection d'Accidents", use_container_width=True, key="nav_detection"):
            st.session_state["active_section"] = "detection"
            st.rerun()
        if st.button("  Prédiction de Risque", use_container_width=True, key="nav_prediction"):
            st.session_state["active_section"] = "prediction"
            st.rerun()
        if st.button("  Carte de Risque", use_container_width=True, key="nav_map"):
            st.session_state["active_section"] = "map"
            st.rerun()

    active = st.session_state["active_section"]
    section_label = (
        "Détection d'Accidents" if active == "detection"
        else "Prédiction de Risque" if active == "prediction"
        else "Carte de Risque"
    )

    st.markdown(
        f"""
        <div class="cyber-header">
            <h1>Road <span class="accent">Accident</span> AI</h1>
            <p>[ {section_label.upper()} ] — Système de détection et prédiction par deep learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if active == "detection":
        render_detection_section()
    elif active == "prediction":
        render_prediction_section()
    elif active == "map":
        render_map_section()


if __name__ == "__main__":
    main()