#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SceneDetector Modul - Erkennt Szenen in Videos mit PySceneDetect
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

# PySceneDetect importieren
from scenedetect import detect, ContentDetector, AdaptiveDetector
from scenedetect import SceneManager, VideoStream
from scenedetect import video_splitter
from scenedetect.scene_manager import SceneList

logger = logging.getLogger(__name__)


class SceneDetector:
    """Klasse zur Erkennung von Szenen in Videos mit PySceneDetect."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialisiert den SceneDetector.
        
        Args:
            input_dir: Verzeichnis mit den Eingangsvideos
            output_dir: Verzeichnis für die erkannten Szenen
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SceneDetector initialisiert mit input_dir={input_dir} und output_dir={output_dir}")
    
    def detect_scenes(self, video_path: str, threshold: float = 27.0,
                      use_adaptive_detector: bool = True) -> SceneList:
        """
        Erkennt Szenen in einem Video.
        
        Args:
            video_path: Pfad zum Video
            threshold: Schwellenwert für die Szenenerkennung
            use_adaptive_detector: Ob der AdaptiveDetector verwendet werden soll
            
        Returns:
            Liste der erkannten Szenen
        """
        logger.info(f"Erkenne Szenen in {video_path} mit threshold={threshold}")
        
        # Wähle den Detektor
        if use_adaptive_detector:
            detector = AdaptiveDetector()
        else:
            detector = ContentDetector(threshold=threshold)
        
        # Erkenne Szenen
        scene_list = detect(video_path, detector)
        
        logger.info(f"{len(scene_list)} Szenen in {video_path} erkannt")
        return scene_list
    
    def split_video(self, video_path: str, scene_list: SceneList, 
                   output_prefix: Optional[str] = None) -> List[str]:
        """
        Teilt ein Video in Szenen auf.
        
        Args:
            video_path: Pfad zum Video
            scene_list: Liste der erkannten Szenen
            output_prefix: Präfix für die Ausgabedateien (Optional)
            
        Returns:
            Liste der Pfade zu den gesplitteten Szenen
        """
        if output_prefix is None:
            # Verwende den Videonamen als Präfix
            output_prefix = Path(video_path).stem
        
        output_path = self.output_dir / output_prefix
        
        logger.info(f"Teile Video {video_path} in {len(scene_list)} Szenen auf")
        
        # Stelle sicher, dass das Ausgabeverzeichnis existiert
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Teile das Video auf
        output_files = video_splitter.split_video_ffmpeg(
            video_path, 
            scene_list, 
            output_path, 
            show_progress=True
        )
        
        logger.info(f"Video in {len(output_files)} Dateien aufgeteilt")
        return output_files
    
    def process_video(self, video_path: str, threshold: float = 27.0,
                     use_adaptive_detector: bool = True) -> List[str]:
        """
        Verarbeitet ein Video vollständig: Erkennt Szenen und teilt es auf.
        
        Args:
            video_path: Pfad zum Video
            threshold: Schwellenwert für die Szenenerkennung
            use_adaptive_detector: Ob der AdaptiveDetector verwendet werden soll
            
        Returns:
            Liste der Pfade zu den gesplitteten Szenen
        """
        # Erkenne Szenen
        scene_list = self.detect_scenes(video_path, threshold, use_adaptive_detector)
        
        # Teile Video in Szenen auf
        output_files = self.split_video(video_path, scene_list)
        
        return output_files
    
    def process_directory(self, threshold: float = 27.0, 
                         use_adaptive_detector: bool = True) -> List[str]:
        """
        Verarbeitet alle Videos im Eingabeverzeichnis.
        
        Args:
            threshold: Schwellenwert für die Szenenerkennung
            use_adaptive_detector: Ob der AdaptiveDetector verwendet werden soll
            
        Returns:
            Liste aller erzeugten Szenendateien
        """
        all_output_files = []
        
        # Unterstützte Videoformate
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        
        # Finde alle Videos im Eingabeverzeichnis
        for video_file in self.input_dir.glob('**/*'):
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                try:
                    output_files = self.process_video(
                        str(video_file), 
                        threshold, 
                        use_adaptive_detector
                    )
                    all_output_files.extend(output_files)
                except Exception as e:
                    logger.error(f"Fehler bei der Verarbeitung von {video_file}: {e}")
        
        logger.info(f"Insgesamt {len(all_output_files)} Szenen aus allen Videos erzeugt")
        return all_output_files


if __name__ == "__main__":
    # Beispiel für die Verwendung des SceneDetectors
    import sys
    from dotenv import load_dotenv
    
    # Lade Umgebungsvariablen
    load_dotenv()
    
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Eingabe- und Ausgabeverzeichnis aus .env oder Standardwerte
    input_dir = os.getenv('VIDEO_INPUT_DIR', 'data/raw_videos')
    output_dir = os.getenv('SCENE_OUTPUT_DIR', 'data/scenes')
    
    detector = SceneDetector(input_dir, output_dir)
    
    if len(sys.argv) > 1:
        # Verarbeite einen bestimmten Film
        video_path = sys.argv[1]
        detector.process_video(video_path)
    else:
        # Verarbeite alle Videos im Eingabeverzeichnis
        detector.process_directory() 