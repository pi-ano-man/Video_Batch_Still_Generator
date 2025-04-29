# Video Batch Still Generator

**Ziel und Zweck des Projekts**
Dieses Projekt automatisiert die Verarbeitung großer Mengen von Video-Dateien, um Szenen verlustfrei zu trennen, Standbilder einer bestimmten Person vorauszuwählen und diese qualitativ auszuwählen. Zur robusten Erkennung der Zielperson wird ein Template-Ansatz mit mehreren Referenzbildern verwendet. Abschließend werden die ausgewählten Bilder in einem letzten Schritt per AI-gestützter Super-Resolution optimiert und exportiert.

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/pi-ano-man/Video_Batch_Still_Generator/ci.yml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![License](https://img.shields.io/github/license/pi-ano-man/Video_Batch_Still_Generator)

## Inhaltsverzeichnis

- [Überblick](#überblick)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Architektur](#architektur)
- [Hauptfunktionen](#hauptfunktionen)
- [Referenzbilder](#referenzbilder)
- [Entwicklung](#entwicklung)
- [Tests](#tests)
- [Lizenz](#lizenz)

## Überblick

Der Video Batch Still Generator ist ein leistungsstarkes Tool zur automatisierten Verarbeitung von Videosammlungen, um hochwertige Standbilder einer bestimmten Person zu extrahieren. Die Hauptfunktionen umfassen:

1. **Automatische Szenenerkennung** - Erkennt Schnitte und Übergänge in Videos
2. **Frame-Extraktion** - Extrahiert einzelne Frames aus erkannten Szenen
3. **Template-basierte Gesichtserkennung** - Filtert Frames basierend auf Referenzbildern
4. **Qualitätsbewertung und Clustering** - Bewertet Bildqualität (Schärfe, Bewegungsunschärfe) und gruppiert ähnliche Bilder
5. **Super-Resolution-Export** - Verbessert die ausgewählten Bilder mit KI-basierter Super-Resolution

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- FFmpeg (für Videoverarbeitung)
- CUDA-fähige GPU (optional, für beschleunigte Super-Resolution)

### Einrichtung

1. Repository klonen:
   ```bash
   git clone https://github.com/pi-ano-man/Video_Batch_Still_Generator.git
   cd Video_Batch_Still_Generator
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unter Windows: venv\Scripts\activate
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Umgebungsvariablen konfigurieren:
   Kopiere die `.env.example`-Datei zu `.env` und passe die Einstellungen nach Bedarf an.

## Verwendung

### Über die Kommandozeile

```bash
# Vollständige Pipeline ausführen
python -m src.main -i path/to/videos -p "Ziel-Person"

# Nur bestimmte Schritte ausführen
python -m src.main -i path/to/videos -p "Ziel-Person" scenes  # Nur Szenenerkennung
python -m src.main -i path/to/videos -p "Ziel-Person" frames  # Nur Frame-Extraktion
python -m src.main -i path/to/videos -p "Ziel-Person" faces   # Nur Gesichtserkennung
python -m src.main -i path/to/videos -p "Ziel-Person" quality # Nur Qualitätsbewertung
python -m src.main -i path/to/videos -p "Ziel-Person" superres # Nur Super-Resolution
```

### Parameter

- `-i, --input`: Eingabedatei oder -verzeichnis
- `-p, --person`: Name der Zielperson
- `--fps`: Frames pro Sekunde für die Extraktion (Standard: 1.0)
- `--format`: Ausgabeformat (png, tiff, jpg) (Standard: png)

### Als Python-Modul

```python
from src.main import BatchStillGenerator

generator = BatchStillGenerator(
    video_input_dir="data/raw_videos",
    scene_output_dir="data/scenes",
    frame_output_dir="data/frames",
    candidate_dir="data/candidates",
    reference_dir="data/reference",
    final_output_dir="data/output",
    face_match_threshold=0.6,
    sr_model="ESRGAN",
    sr_scale=2
)

# Vollständige Pipeline für ein einzelnes Video ausführen
results = generator.process_video(
    video_path="path/to/video.mp4",
    target_person="Ziel-Person",
    fps=1.0,
    output_format="png"
)

# Vollständige Pipeline für ein Verzeichnis ausführen
results = generator.process_directory(
    target_person="Ziel-Person",
    fps=1.0,
    output_format="png"
)
```

## Architektur

Der Generator besteht aus mehreren spezialisierten Modulen:

1. **SceneDetector** - Erkennt Szenenübergänge in Videos mit PySceneDetect
2. **FrameExtractor** - Extrahiert Frames mit FFmpeg
3. **FaceFilter** - Template-basierte Gesichtserkennung mit mehreren Referenzbildern
4. **QualityEvaluator** - Bewertet Bildqualität und clustert ähnliche Bilder
5. **SuperResolutionExporter** - Wendet KI-basierte Super-Resolution an

Die Module arbeiten in einer Pipeline zusammen und jeder Schritt kann auch einzeln ausgeführt werden.

## Hauptfunktionen

### Szenenerkennung

Die Szenenerkennung verwendet PySceneDetect mit ContentDetector und AdaptiveDetector, um Szenenübergänge in Videos zu erkennen.

```python
# Szenen in einem Video erkennen
scene_paths = generator.detect_scenes("video.mp4")
```

### Frame-Extraktion

Die Frame-Extraktion verwendet FFmpeg, um einzelne Frames aus erkannten Szenen zu extrahieren.

```python
# Frames aus einer Szene extrahieren
frames = generator.extract_frames("scene.mp4", fps=1.0)
```

### Gesichtserkennung

Die Gesichtserkennung verwendet einen Template-Ansatz mit mehreren Referenzbildern, um Frames zu filtern, die die Zielperson enthalten.

```python
# Gesichter in Frames filtern
results = generator.filter_faces(frames, target_person="Ziel-Person")
```

### Qualitätsbewertung

Die Qualitätsbewertung bewertet die Bildqualität (Schärfe, Bewegungsunschärfe) und clustert ähnliche Bilder, um die besten Repräsentanten auszuwählen.

```python
# Bildqualität bewerten und clustern
results = generator.evaluate_quality(target_person="Ziel-Person")
```

### Super-Resolution

Der Super-Resolution-Export verwendet Open-Source-Modelle (ESRGAN, EDVR, etc.), um die Qualität der ausgewählten Bilder zu verbessern.

```python
# Super-Resolution anwenden und exportieren
exported = generator.export_superresolution(best_images, output_format="png")
```

## Referenzbilder

Referenzbilder für die Gesichtserkennung sollten im Verzeichnis `data/reference/<Name-der-Person>/` gespeichert werden. Mehrere Bilder pro Person verbessern die Erkennungsgenauigkeit.

Beispiel:
```
data/reference/
  ├── Person1/
  │     ├── bild1.jpg
  │     ├── bild2.jpg
  │     └── bild3.jpg
  └── Person2/
        ├── bild1.jpg
        └── bild2.jpg
```

## Entwicklung

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### Codekonventionen

- Python-Module in **snake_case**
- Python-Klassen in **PascalCase**
- Methoden und Variablen in **snake_case**
- Docstrings im Google-Style
- Zeilenlänge: max. 100 Zeichen
- 4 Spaces für Einrückungen

## Tests

```bash
# Tests ausführen
pytest

# Mit Coverage-Bericht
pytest --cov=src

# Linting
flake8 src

# Sicherheitsanalyse
bandit -r src
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe [LICENSE](LICENSE) für Details.