# 🚨 Road Accident AI System

Système intelligent de **détection** et **prédiction** d'accidents routiers par Deep Learning.

Développé dans le cadre d'un mémoire de Master en Aide à la Décision et Systèmes Intelligents — Université d'Oran 1, 2026.

---

## Fonctionnalités

| Module | Description |
|---|---|
| 🔍 Détection | YOLOv8s — détection d'accidents sur images et vidéos en temps réel |
| 📊 Prédiction | XGBoost — estimation du risque d'accident grave (%) selon les conditions |
| 🗺️ Carte de risque | Heatmap géographique basée sur 1M d'accidents réels (US Accidents DB) |

---

## Installation

```bash
git clone https://github.com/votre-repo/road-accident-ai.git
cd road-accident-ai
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Données requises

### 1. US Accidents Dataset
Télécharger depuis Kaggle :
👉 https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

Placer le fichier dans :
```
data/US_Accidents_March23.csv
```

### 2. Modèle XGBoost
Téléchargez le modèle pré-entraîné et placer dans :
```
models/model_xgboost.pkl
```

### 3. Modèle YOLOv8
Téléchargez`best.pt` depuis repo GitHub et placer dans :
```
models/best.pt
```

---

## Lancement

```bash
streamlit run app.py
```

L'interface est accessible sur `http://localhost:8501`

---

## Structure du projet

```
# Road-accident-prediction-and-detection-system-YOLO

## 📁 Structure du projet

```bash
Road-accident-prediction-and-detection-system-YOLO/
│
├── examples/              # Exemples d'utilisation
│
├── models/                # Modèles YOLO et Xgboost
│
├── runs/                  # Résultats d'entraînement
│
├── src/                   
│   ├── app.py             # Application principale
│   ├── model.py           # Gestion du modèle YOLO
│   └── train.py           # Entraînement du modèle
│
├── Prediction/            # Résultats des prédictions
│
├── README.md              # Documentation
│
└── requirements.txt       # Dépendances Python

## Modèles

| Modèle | Architecture | Dataset | Métriques |
|---|---|---|---|
| Détection | YOLOv8s | 4 250 images (Roboflow) | Precision 79% — Recall: 60.2% |
| Prédiction | XGBoost | 918 431 accidents (US) | AUC-ROC: 0.72 |

---

## Auteurs

- **Sersar Mohammed El Amine**
- **Mesbah Abdelmajid Ryad**

Université d'Oran 1 — Faculté des Sciences Exactes et Appliquées — 2026