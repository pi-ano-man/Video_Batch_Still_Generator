#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hauptmodul für Video Batch Still Generator.
Führt die gesamte Pipeline von der Videoanalyse bis zum Super-Resolution-Export aus.
"""

import os
import logging
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dotenv import load_dotenv

# Import der Projektmodule
from scene_detector.scene_detector import SceneDetector
from frame_extractor.frame_extractor import FrameExtractor
from face_filter.face_filter import FaceFilter
from quality_evaluator.quality_evaluator import QualityEvaluator
from superresolution_exporter.superresolution_exporter import SuperResolutionExporter, SRModel

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class BatchStillGenerator:
    """
    Hauptklasse, die alle Module zusammenbringt und den gesamten Workflow steuert.
    """
    
    def __init__(self, 
                video_input_dir: str,
                scene_output_dir: str,
                frame_output_dir: str,
                candidate_dir: str,
                reference_dir: str,
                final_output_dir: str,
                face_match_threshold: float = 0.6,
                sr_model: str = "ESRGAN",
                sr_scale: int = 2,
                sr_cli_path: Optional[str] = None):
        """
        Initialisiert den BatchStillGenerator.
        
        Args:
            video_input_dir: Verzeichnis mit den Eingangsvideos
            scene_output_dir: Verzeichnis für erkannte Szenen
            frame_output_dir: Verzeichnis für extrahierte Frames
            candidate_dir: Verzeichnis für Kandidaten-Frames (gefiltert)
            reference_dir: Verzeichnis mit Referenzbildern
            final_output_dir: Verzeichnis für die finalen Ergebnisse
            face_match_threshold: Schwellenwert für die Gesichtserkennung
            sr_model: Super-Resolution-Modell (ESRGAN, EDVR, etc.)
            sr_scale: Skalierungsfaktor für Super-Resolution
            sr_cli_path: Pfad zum externen CLI-Tool (optional)
        """
        self.video_input_dir = video_input_dir
        self.scene_output_dir = scene_output_dir
        self.frame_output_dir = frame_output_dir
        self.candidate_dir = candidate_dir
        self.reference_dir = reference_dir
        self.final_output_dir = final_output_dir
        
        # Parameter für Module
        self.face_match_threshold = face_match_threshold
        self.sr_model = sr_model
        self.sr_scale = sr_scale
        self.sr_cli_path = sr_cli_path
        
        # Initialisiere die Modulinstanzen, aber erstelle sie erst bei Bedarf
        self._scene_detector = None
        self._frame_extractor = None
        self._face_filter = None
        self._quality_evaluator = None
        self._sr_exporter = None
        
        # Erstelle Verzeichnisse, falls sie nicht existieren
        for directory in [video_input_dir, scene_output_dir, frame_output_dir, 
                         candidate_dir, reference_dir, final_output_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info("BatchStillGenerator initialisiert")
    
    @property
    def scene_detector(self) -> SceneDetector:
        """Getter für den SceneDetector."""
        if self._scene_detector is None:
            self._scene_detector = SceneDetector(
                input_dir=self.video_input_dir,
                output_dir=self.scene_output_dir
            )
        return self._scene_detector
    
    @property
    def frame_extractor(self) -> FrameExtractor:
        """Getter für den FrameExtractor."""
        if self._frame_extractor is None:
            self._frame_extractor = FrameExtractor(
                input_dir=self.scene_output_dir,
                output_dir=self.frame_output_dir
            )
        return self._frame_extractor
    
    @property
    def face_filter(self) -> FaceFilter:
        """Getter für den FaceFilter."""
        if self._face_filter is None:
            self._face_filter = FaceFilter(
                reference_dir=self.reference_dir,
                output_dir=self.candidate_dir,
                face_match_threshold=self.face_match_threshold
            )
        return self._face_filter
    
    @property
    def quality_evaluator(self) -> QualityEvaluator:
        """Getter für den QualityEvaluator."""
        if self._quality_evaluator is None:
            self._quality_evaluator = QualityEvaluator(
                input_dir=self.candidate_dir
            )
        return self._quality_evaluator
    
    @property
    def sr_exporter(self) -> SuperResolutionExporter:
        """Getter für den SuperResolutionExporter."""
        if self._sr_exporter is None:
            self._sr_exporter = SuperResolutionExporter(
                input_dir=self.candidate_dir,
                output_dir=self.final_output_dir,
                model=self.sr_model,
                scale=self.sr_scale,
                cli_path=self.sr_cli_path
            )
        return self._sr_exporter
    
    def detect_scenes(self, video_path: Optional[str] = None,
                    use_adaptive_detector: bool = True) -> List[str]:
        """
        Erkennt Szenen in Videos.
        
        Args:
            video_path: Pfad zu einem einzelnen Video (optional)
            use_adaptive_detector: Ob der adaptive Detektor verwendet werden soll
            
        Returns:
            Liste der Pfade zu den erkannten Szenen
        """
        logger.info("Starte Szenenerkennung...")
        
        if video_path:
            # Verarbeite ein einzelnes Video
            return self.scene_detector.process_video(
                video_path=video_path,
                use_adaptive_detector=use_adaptive_detector
            )
        else:
            # Verarbeite alle Videos im Eingabeverzeichnis
            return self.scene_detector.process_directory(
                use_adaptive_detector=use_adaptive_detector
            )
    
    def extract_frames(self, scene_path: Optional[str] = None,
                     fps: float = 1.0) -> Dict[str, List[str]]:
        """
        Extrahiert Frames aus Szenen.
        
        Args:
            scene_path: Pfad zu einer einzelnen Szene (optional)
            fps: Frames pro Sekunde
            
        Returns:
            Dictionary mit Szenenpfaden und Listen von Frame-Pfaden
        """
        logger.info(f"Starte Frame-Extraktion mit fps={fps}...")
        
        if scene_path:
            # Verarbeite eine einzelne Szene
            frames = self.frame_extractor.extract_frames(
                video_path=scene_path,
                fps=fps
            )
            return {scene_path: frames} if frames else {}
        else:
            # Verarbeite alle Szenen im Verzeichnis
            return self.frame_extractor.process_directory(fps=fps)
    
    def filter_faces(self, frames: List[str], target_person: Optional[str] = None) -> Dict[str, Dict]:
        """
        Filtert Frames nach Gesichtserkennung.
        
        Args:
            frames: Liste von Frame-Pfaden
            target_person: Name der Zielperson (optional)
            
        Returns:
            Dictionary mit Filter-Ergebnissen
        """
        logger.info(f"Starte Gesichtsfilterung für {len(frames)} Frames...")
        
        return self.face_filter.filter_frames(
            frames=frames,
            target_person=target_person
        )
    
    def evaluate_quality(self, target_person: Optional[str] = None,
                        face_regions: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Bewertet die Qualität der Kandidaten-Frames und clustert ähnliche Bilder.
        
        Args:
            target_person: Name der Zielperson (optional)
            face_regions: Gesichtskoordinaten (optional)
            
        Returns:
            Dictionary mit Qualitätsbewertungen und besten Bildern
        """
        logger.info("Starte Qualitätsbewertung und Clustering...")
        
        return self.quality_evaluator.process_directory(
            target_person=target_person,
            face_regions=face_regions
        )
    
    def export_superresolution(self, image_paths: Optional[List[str]] = None,
                            output_format: str = "png") -> List[str]:
        """
        Führt Super-Resolution und Export für Bilder durch.
        
        Args:
            image_paths: Liste von Bildpfaden (optional, sonst alle im Verzeichnis)
            output_format: Ausgabeformat (png oder tiff)
            
        Returns:
            Liste der Pfade zu den hochskalierten Bildern
        """
        logger.info("Starte Super-Resolution und Export...")
        
        if image_paths:
            # Verarbeite bestimmte Bilder
            return self.sr_exporter.batch_upscale(
                image_paths=image_paths,
                output_format=output_format
            )
        else:
            # Verarbeite alle Bilder im Verzeichnis
            return self.sr_exporter.process_directory(
                output_format=output_format
            )
    
    def process_video(self, video_path: str, target_person: Optional[str] = None,
                    fps: float = 1.0, output_format: str = "png") -> Dict[str, Any]:
        """
        Verarbeitet ein einzelnes Video durch die gesamte Pipeline.
        
        Args:
            video_path: Pfad zum Video
            target_person: Name der Zielperson (optional)
            fps: Frames pro Sekunde für die Frame-Extraktion
            output_format: Ausgabeformat für den Export
            
        Returns:
            Dictionary mit Ergebnissen
        """
        logger.info(f"Starte vollständige Verarbeitung von {video_path}...")
        start_time = time.time()
        
        # 1. Szenen erkennen
        scene_paths = self.detect_scenes(video_path)
        if not scene_paths:
            logger.error(f"Keine Szenen in {video_path} erkannt")
            return {"error": "Keine Szenen erkannt", "video_path": video_path}
        
        # 2. Frames extrahieren
        all_frames = []
        for scene_path in scene_paths:
            frames_dict = self.extract_frames(scene_path, fps)
            frames = frames_dict.get(scene_path, [])
            all_frames.extend(frames)
        
        if not all_frames:
            logger.error(f"Keine Frames aus {video_path} extrahiert")
            return {"error": "Keine Frames extrahiert", "video_path": video_path, "scenes": scene_paths}
        
        # 3. Gesichter filtern
        filter_results = self.filter_faces(all_frames, target_person)
        
        # Extrahiere übereinstimmende Frames und Gesichtsregionen
        matches = [path for path, result in filter_results.items() if result.get("match", False)]
        face_regions = {path: result.get("face_region", {}) for path, result in filter_results.items() if "face_region" in result}
        
        if not matches:
            logger.error(f"Keine übereinstimmenden Gesichter in {video_path} gefunden")
            return {
                "error": "Keine übereinstimmenden Gesichter",
                "video_path": video_path,
                "scenes": scene_paths,
                "frames": all_frames
            }
        
        # 4. Qualität bewerten und clustern
        quality_results = self.evaluate_quality(target_person, face_regions)
        best_images = quality_results.get("best_images", [])
        
        if not best_images:
            logger.error(f"Keine besten Bilder aus {video_path} ausgewählt")
            return {
                "error": "Keine besten Bilder ausgewählt",
                "video_path": video_path,
                "scenes": scene_paths,
                "frames": all_frames,
                "matches": matches
            }
        
        # 5. Super-Resolution und Export
        exported_images = self.export_superresolution(best_images, output_format)
        
        # Berechne Laufzeit
        elapsed_time = time.time() - start_time
        
        # Erstelle Ergebnisdictionary
        results = {
            "video_path": video_path,
            "scenes": scene_paths,
            "frames": all_frames,
            "matches": matches,
            "best_images": best_images,
            "exported_images": exported_images,
            "elapsed_time": elapsed_time
        }
        
        logger.info(f"Verarbeitung von {video_path} abgeschlossen in {elapsed_time:.2f} Sekunden")
        logger.info(f"Ergebnisse: {len(scene_paths)} Szenen, {len(all_frames)} Frames, " +
                  f"{len(matches)} Matches, {len(best_images)} beste Bilder, " +
                  f"{len(exported_images)} exportierte Bilder")
        
        return results
    
    def process_directory(self, target_person: Optional[str] = None,
                        fps: float = 1.0, output_format: str = "png") -> Dict[str, Dict[str, Any]]:
        """
        Verarbeitet alle Videos im Eingabeverzeichnis durch die gesamte Pipeline.
        
        Args:
            target_person: Name der Zielperson (optional)
            fps: Frames pro Sekunde für die Frame-Extraktion
            output_format: Ausgabeformat für den Export
            
        Returns:
            Dictionary mit Videodateien als Schlüsseln und Ergebnissen als Werten
        """
        logger.info(f"Starte Batch-Verarbeitung aller Videos in {self.video_input_dir}...")
        start_time = time.time()
        
        # Finde alle Videos im Eingabeverzeichnis
        supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
        video_paths = []
        
        for fmt in supported_formats:
            video_paths.extend([str(p) for p in Path(self.video_input_dir).glob(f"**/*{fmt}")])
        
        if not video_paths:
            logger.error(f"Keine Videos in {self.video_input_dir} gefunden")
            return {}
        
        logger.info(f"{len(video_paths)} Videos gefunden")
        
        # Verarbeite jedes Video
        results = {}
        for i, video_path in enumerate(video_paths):
            logger.info(f"Verarbeite Video {i+1}/{len(video_paths)}: {video_path}")
            try:
                video_result = self.process_video(
                    video_path=video_path,
                    target_person=target_person,
                    fps=fps,
                    output_format=output_format
                )
                results[video_path] = video_result
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung von {video_path}: {e}")
                results[video_path] = {"error": str(e), "video_path": video_path}
        
        # Berechne Gesamtlaufzeit
        elapsed_time = time.time() - start_time
        
        # Erstelle Zusammenfassung
        total_scenes = sum(len(result.get("scenes", [])) for result in results.values())
        total_frames = sum(len(result.get("frames", [])) for result in results.values())
        total_matches = sum(len(result.get("matches", [])) for result in results.values())
        total_best = sum(len(result.get("best_images", [])) for result in results.values())
        total_exported = sum(len(result.get("exported_images", [])) for result in results.values())
        
        logger.info(f"Batch-Verarbeitung abgeschlossen in {elapsed_time:.2f} Sekunden")
        logger.info(f"Gesamtergebnisse: {len(video_paths)} Videos, {total_scenes} Szenen, " +
                   f"{total_frames} Frames, {total_matches} Matches, " +
                   f"{total_best} beste Bilder, {total_exported} exportierte Bilder")
        
        # Füge Zusammenfassung zu den Ergebnissen hinzu
        results["summary"] = {
            "total_videos": len(video_paths),
            "total_scenes": total_scenes,
            "total_frames": total_frames,
            "total_matches": total_matches,
            "total_best_images": total_best,
            "total_exported_images": total_exported,
            "elapsed_time": elapsed_time
        }
        
        return results


def load_env_or_default(key: str, default: Any) -> Any:
    """Lädt einen Wert aus der .env-Datei oder gibt einen Standardwert zurück."""
    value = os.getenv(key)
    if value is None:
        return default
    return value


def main():
    """Hauptfunktion für die Kommandozeilenschnittstelle."""
    # Lade Umgebungsvariablen
    load_dotenv()
    
    # Parse Kommandozeilenargumente
    parser = argparse.ArgumentParser(description="Video Batch Still Generator")
    
    # Allgemeine Argumente
    parser.add_argument("-i", "--input", help="Eingabedatei oder -verzeichnis", 
                      default=os.getenv("VIDEO_INPUT_DIR", "data/raw_videos"))
    parser.add_argument("-p", "--person", help="Name der Zielperson")
    parser.add_argument("--fps", help="Frames pro Sekunde für die Extraktion", 
                      type=float, default=1.0)
    parser.add_argument("--format", help="Ausgabeformat (png oder tiff)", 
                      choices=["png", "tiff", "jpg"], default="png")
    
    # Subparsers für verschiedene Modi
    subparsers = parser.add_subparsers(dest="mode", help="Betriebsmodus")
    
    # Szenenerkennung
    scenes_parser = subparsers.add_parser("scenes", help="Nur Szenenerkennung durchführen")
    scenes_parser.add_argument("--adaptive", help="Adapativen Detektor verwenden", 
                            action="store_true", default=True)
    
    # Frame-Extraktion
    frames_parser = subparsers.add_parser("frames", help="Nur Frame-Extraktion durchführen")
    
    # Gesichtsfilterung
    faces_parser = subparsers.add_parser("faces", help="Nur Gesichtsfilterung durchführen")
    faces_parser.add_argument("--threshold", help="Schwellenwert für die Gesichtserkennung", 
                           type=float, default=float(os.getenv("FACE_MATCH_THRESHOLD", "0.6")))
    
    # Qualitätsbewertung
    quality_parser = subparsers.add_parser("quality", help="Nur Qualitätsbewertung durchführen")
    
    # Super-Resolution
    sr_parser = subparsers.add_parser("superres", help="Nur Super-Resolution durchführen")
    sr_parser.add_argument("--model", help="Super-Resolution-Modell", 
                        default=os.getenv("SR_MODEL", "ESRGAN"))
    sr_parser.add_argument("--scale", help="Skalierungsfaktor", 
                        type=int, default=int(os.getenv("SR_SCALE", "2")))
    
    # Vollständige Pipeline
    full_parser = subparsers.add_parser("full", help="Vollständige Pipeline durchführen")
    
    # Parse Argumente
    args = parser.parse_args()
    
    # Lade Konfiguration aus .env oder Standardwerte
    config = {
        "video_input_dir": os.getenv("VIDEO_INPUT_DIR", "data/raw_videos"),
        "scene_output_dir": os.getenv("SCENE_OUTPUT_DIR", "data/scenes"),
        "frame_output_dir": os.getenv("FRAME_OUTPUT_DIR", "data/frames"),
        "candidate_dir": os.getenv("CANDIDATE_DIR", "data/candidates"),
        "reference_dir": os.getenv("REFERENCE_IMAGES_DIR", "data/reference"),
        "final_output_dir": os.getenv("FINAL_OUTPUT_DIR", "data/output"),
        "face_match_threshold": float(os.getenv("FACE_MATCH_THRESHOLD", "0.6")),
        "sr_model": os.getenv("SR_MODEL", "ESRGAN"),
        "sr_scale": int(os.getenv("SR_SCALE", "2")),
        "sr_cli_path": os.getenv("SR_CLI_PATH")
    }
    
    # Überschreibe Konfiguration mit Kommandozeilenargumenten
    if args.input:
        if os.path.isfile(args.input):
            # Wenn die Eingabe eine Datei ist, setze den Eingabepfad für die Verarbeitung
            input_file = args.input
            config["video_input_dir"] = str(Path(args.input).parent)
        else:
            # Wenn die Eingabe ein Verzeichnis ist, setze das Eingabeverzeichnis
            config["video_input_dir"] = args.input
            input_file = None
    else:
        input_file = None
    
    # Initialisiere den Generator
    generator = BatchStillGenerator(**config)
    
    # Führe je nach Modus unterschiedliche Aktionen aus
    if args.mode == "scenes":
        # Nur Szenen erkennen
        if input_file:
            scenes = generator.detect_scenes(input_file, args.adaptive)
            print(f"{len(scenes)} Szenen erkannt in {input_file}")
        else:
            scenes = generator.detect_scenes(use_adaptive_detector=args.adaptive)
            print(f"{len(scenes)} Szenen insgesamt erkannt")
            
    elif args.mode == "frames":
        # Nur Frames extrahieren
        if input_file:
            frames_dict = generator.extract_frames(input_file, args.fps)
            total_frames = sum(len(frames) for frames in frames_dict.values())
            print(f"{total_frames} Frames extrahiert aus {input_file}")
        else:
            frames_dict = generator.extract_frames(fps=args.fps)
            total_frames = sum(len(frames) for frames in frames_dict.values())
            print(f"{total_frames} Frames insgesamt extrahiert")
            
    elif args.mode == "faces":
        # Nur Gesichter filtern
        if args.threshold:
            generator.face_match_threshold = args.threshold
        
        # Sammle alle Frames
        frames = []
        if input_file:
            # Wenn input_file ein Verzeichnis mit Frames ist
            input_path = Path(input_file)
            if input_path.is_dir():
                supported_formats = ('*.jpg', '*.jpeg', '*.png')
                for fmt in supported_formats:
                    frames.extend([str(p) for p in input_path.glob(f"**/{fmt}")])
            else:
                # Extrahiere Frames aus dem Video
                frames_dict = generator.extract_frames(input_file, args.fps)
                for frame_list in frames_dict.values():
                    frames.extend(frame_list)
        else:
            # Sammle Frames aus dem Frame-Verzeichnis
            frame_dir = Path(config["frame_output_dir"])
            supported_formats = ('*.jpg', '*.jpeg', '*.png')
            for fmt in supported_formats:
                frames.extend([str(p) for p in frame_dir.glob(f"**/{fmt}")])
                
        # Filtere Gesichter
        if frames:
            results = generator.filter_faces(frames, args.person)
            matches = [path for path, result in results.items() if result.get("match", False)]
            print(f"{len(matches)}/{len(frames)} Frames mit übereinstimmenden Gesichtern gefunden")
        else:
            print("Keine Frames zum Filtern gefunden")
            
    elif args.mode == "quality":
        # Nur Qualität bewerten
        results = generator.evaluate_quality(args.person)
        best_images = results.get("best_images", [])
        print(f"{len(best_images)} beste Bilder ausgewählt")
        for i, path in enumerate(best_images):
            score = results["quality_scores"][path]["overall_score"]
            print(f"{i+1}. {path} (Score: {score:.2f})")
            
    elif args.mode == "superres":
        # Nur Super-Resolution durchführen
        if args.model:
            generator.sr_model = args.model
        if args.scale:
            generator.sr_scale = args.scale
            
        if input_file:
            # Wenn input_file ein Verzeichnis mit Bildern ist
            input_path = Path(input_file)
            if input_path.is_dir():
                generator.sr_exporter.input_dir = input_path
                exported = generator.export_superresolution(output_format=args.format)
                print(f"{len(exported)} Bilder hochskaliert")
            else:
                # Einzelnes Bild verarbeiten
                exported = generator.export_superresolution([input_file], args.format)
                print(f"{len(exported)} Bilder hochskaliert")
        else:
            # Verarbeite alle Bilder im Kandidatenverzeichnis
            exported = generator.export_superresolution(output_format=args.format)
            print(f"{len(exported)} Bilder hochskaliert")
            
    else:  # "full" oder kein Modus angegeben
        # Vollständige Pipeline
        if input_file:
            # Verarbeite ein einzelnes Video
            results = generator.process_video(
                video_path=input_file,
                target_person=args.person,
                fps=args.fps,
                output_format=args.format
            )
            
            # Zeige Zusammenfassung
            if "error" in results:
                print(f"Fehler: {results['error']}")
            else:
                print(f"Ergebnisse für {input_file}:")
                print(f"  Szenen: {len(results['scenes'])}")
                print(f"  Frames: {len(results['frames'])}")
                print(f"  Matches: {len(results['matches'])}")
                print(f"  Beste Bilder: {len(results['best_images'])}")
                print(f"  Exportierte Bilder: {len(results['exported_images'])}")
                print(f"  Laufzeit: {results['elapsed_time']:.2f} Sekunden")
        else:
            # Verarbeite alle Videos im Verzeichnis
            results = generator.process_directory(
                target_person=args.person,
                fps=args.fps,
                output_format=args.format
            )
            
            # Zeige Zusammenfassung
            if "summary" in results:
                summary = results["summary"]
                print("Zusammenfassung der Batch-Verarbeitung:")
                print(f"  Videos: {summary['total_videos']}")
                print(f"  Szenen: {summary['total_scenes']}")
                print(f"  Frames: {summary['total_frames']}")
                print(f"  Matches: {summary['total_matches']}")
                print(f"  Beste Bilder: {summary['total_best_images']}")
                print(f"  Exportierte Bilder: {summary['total_exported_images']}")
                print(f"  Gesamtlaufzeit: {summary['elapsed_time']:.2f} Sekunden")


if __name__ == "__main__":
    main() 