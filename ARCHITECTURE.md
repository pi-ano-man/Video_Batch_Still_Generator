# Architekturübersicht

## Komponenten
1. **SceneDetector** (Python) – Szenenerkennung via PySceneDetect
2. **FrameExtractor** (Python + FFmpeg) – Frame-Extraktion mit VideoToolbox
3. **FaceFilter** (Python) – Template-basierte Erkennung: Lade mehrere Referenzbilder, generiere Encodings und vergleiche Video-Frame-Encodings per Distanzschwelle
4. **QualityEvaluator** (Python) – Schärfeanalyse, Bewegungsunschärfe, Clustering ähnlicher Frames
5. **SelectionController** (Python) – Zwischenspeicherung, Benutzer-Auswahl, Feinjustage
6. **SuperResolutionExporter** (Python / CLI) – Finaler Schritt: AI-gestützte Super-Resolution und Batch-Export

## Datenfluss
Raw Videos → SceneDetector → Scenes
Scenes → FrameExtractor → Frames
Frames → FaceFilter (mehrere Referenz-Encodings) → Candidate Frames
Candidate Frames → QualityEvaluator → Top Candidates
Top Candidates → SelectionController ↔ GUI
Ausgewählte Bilder → SuperResolutionExporter → Final Output

## Schnittstellen
- **REST API** (intern) zwischen GUI und Backend
- **Dateisystem** im `/data`-Workspace:
  - `/data/raw_videos/`
  - `/data/scenes/`
  - `/data/frames/`
  - `/data/candidates/`
  - `/data/output/`
  - **`/data/reference/`** – Ordner mit Referenzbildern für jede Zielperson

  ## 4. .env.example
```env
# Git-Repository
AUTH_GIT_URL=https://github.com/pi-ano-man/Video_Batch_Still_Generator

# Arbeitsverzeichnisse
VIDEO_INPUT_DIR=/data/raw_videos
SCENE_OUTPUT_DIR=/data/scenes
FRAME_OUTPUT_DIR=/data/frames
CANDIDATE_DIR=/data/candidates
FINAL_OUTPUT_DIR=/data/output
REFERENCE_IMAGES_DIR=/data/reference

# Gesichtserkennung
TARGET_FACE_THRESHOLD=0.6

# Super-Resolution-Export (letzter Schritt)
# lokal per CLI-Tool oder Python-Wrapper
# SR_CLI_PATH=/usr/local/bin/sr_exporter
# SR_MODEL=EDVR
# SR_SCALE=2

# QML-Einstellungen
QML_MAIN_FILE=src/gui/Main.qml

# Logging
LOG_LEVEL=INFO