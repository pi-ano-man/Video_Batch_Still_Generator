#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FrameExtractor Modul - Extrahiert Frames aus Videos mit FFmpeg
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Klasse zur Extraktion von Frames aus Videos mit FFmpeg."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialisiert den FrameExtractor.
        
        Args:
            input_dir: Verzeichnis mit den Eingangsvideos
            output_dir: Verzeichnis für die extrahierten Frames
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FrameExtractor initialisiert mit input_dir={input_dir} und output_dir={output_dir}")
    
    def extract_frames(self, video_path: str, fps: float = 1.0, 
                      output_prefix: Optional[str] = None) -> List[str]:
        """
        Extrahiert Frames aus einem Video mit einer bestimmten Rate.
        
        Args:
            video_path: Pfad zum Video
            fps: Frames pro Sekunde (1.0 = ein Frame pro Sekunde)
            output_prefix: Präfix für die Ausgabedateien (Optional)
            
        Returns:
            Liste der Pfade zu den extrahierten Frames
        """
        if output_prefix is None:
            # Verwende den Videonamen als Präfix
            output_prefix = Path(video_path).stem
        
        # Erstelle Ausgabeverzeichnis für dieses Video
        output_dir = self.output_dir / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ausgabemuster für die Frames
        output_pattern = str(output_dir / f"{output_prefix}_%04d.jpg")
        
        logger.info(f"Extrahiere Frames aus {video_path} mit fps={fps}")
        
        try:
            # Verwende ffmpeg-python für die Frame-Extraktion
            (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=fps)
                .output(output_pattern, q=2)  # q=2 für hohe Qualität
                .run(quiet=True, overwrite_output=True)
            )
            
            # Sammle alle erzeugten Frame-Dateien
            frames = list(output_dir.glob("*.jpg"))
            frames.sort()  # Sortiere Frames nach Namen
            
            logger.info(f"{len(frames)} Frames aus {video_path} extrahiert")
            return [str(f) for f in frames]
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg-Fehler beim Extrahieren von Frames aus {video_path}: {e.stderr.decode()}")
            return []
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren von Frames aus {video_path}: {e}")
            return []
    
    def extract_frames_at_positions(self, video_path: str, 
                                   positions: List[float],
                                   output_prefix: Optional[str] = None) -> List[str]:
        """
        Extrahiert Frames an bestimmten Zeitpositionen.
        
        Args:
            video_path: Pfad zum Video
            positions: Liste von Zeitpositionen in Sekunden
            output_prefix: Präfix für die Ausgabedateien (Optional)
            
        Returns:
            Liste der Pfade zu den extrahierten Frames
        """
        if output_prefix is None:
            # Verwende den Videonamen als Präfix
            output_prefix = Path(video_path).stem
        
        # Erstelle Ausgabeverzeichnis für dieses Video
        output_dir = self.output_dir / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_frames = []
        
        logger.info(f"Extrahiere {len(positions)} spezifische Frames aus {video_path}")
        
        for i, pos in enumerate(positions):
            output_file = output_dir / f"{output_prefix}_{i:04d}_{pos:.2f}s.jpg"
            
            try:
                # Extrahiere ein einzelnes Frame an der Position
                (
                    ffmpeg
                    .input(video_path, ss=pos)  # Setze Startposition
                    .output(str(output_file), vframes=1, q=2)  # Extrahiere genau 1 Frame
                    .run(quiet=True, overwrite_output=True)
                )
                
                if output_file.exists():
                    extracted_frames.append(str(output_file))
                
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg-Fehler beim Extrahieren des Frames bei {pos}s: {e.stderr.decode()}")
            except Exception as e:
                logger.error(f"Fehler beim Extrahieren des Frames bei {pos}s: {e}")
        
        logger.info(f"{len(extracted_frames)} von {len(positions)} Frames erfolgreich extrahiert")
        return extracted_frames
    
    def extract_video_metadata(self, video_path: str) -> Dict[str, Union[int, float, str]]:
        """
        Extrahiert Metadaten aus einem Video (Dauer, FPS, Auflösung, usw.).
        
        Args:
            video_path: Pfad zum Video
            
        Returns:
            Dictionary mit Video-Metadaten
        """
        try:
            # FFmpeg-Probe für Metadaten
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)
            
            if video_stream is None:
                logger.error(f"Kein Video-Stream in {video_path} gefunden")
                return {}
            
            # Extrahiere die wichtigsten Metadaten
            # Berechne fps (kann in verschiedenen Formaten vorliegen)
            fps_str = video_stream.get('r_frame_rate', '0/0')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)
            
            # Extrahiere Dauer
            duration = float(probe.get('format', {}).get('duration', 0))
            
            # Sammle Metadaten
            metadata = {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': fps,
                'duration': duration,
                'total_frames': int(duration * fps) if fps > 0 else 0,
                'codec': video_stream.get('codec_name', 'unknown'),
                'format': probe.get('format', {}).get('format_name', 'unknown')
            }
            
            logger.info(f"Metadaten für {video_path} extrahiert: {metadata}")
            return metadata
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg-Fehler beim Extrahieren der Metadaten aus {video_path}: {e.stderr.decode()}")
            return {}
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren der Metadaten aus {video_path}: {e}")
            return {}
    
    def video_to_numpy(self, video_path: str, start_time: float = 0, 
                      max_frames: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Konvertiert ein Video in ein NumPy-Array für die weitere Verarbeitung.
        
        Args:
            video_path: Pfad zum Video
            start_time: Startzeit in Sekunden
            max_frames: Maximale Anzahl von Frames (Optional)
            
        Returns:
            Tuple aus NumPy-Array und Metadaten
        """
        try:
            # Extrahiere zuerst die Video-Metadaten
            metadata = self.extract_video_metadata(video_path)
            if not metadata:
                logger.error(f"Konnte keine Metadaten für {video_path} extrahieren")
                return np.array([]), {}
            
            # Vorbereiten der FFmpeg-Befehlskette
            input_stream = ffmpeg.input(video_path, ss=start_time)
            
            # Begrenze die Anzahl der Frames, falls angegeben
            if max_frames is not None:
                input_stream = input_stream.trim(start_frame=0, end_frame=max_frames)
                input_stream = input_stream.setpts('PTS-STARTPTS')
            
            # Bereite den Ausgabestream vor
            output_stream = (
                input_stream
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True)
            )
            
            # Lese Ausgabe in ein NumPy-Array
            width = metadata['width']
            height = metadata['height']
            
            # Falls max_frames nicht angegeben ist, verwende eine Schätzung
            if max_frames is None:
                remaining_duration = metadata['duration'] - start_time
                max_frames = int(remaining_duration * metadata['fps'])
            
            # Bereite das NumPy-Array vor
            frames = []
            
            # Lese Frame für Frame
            while len(frames) < max_frames:
                in_bytes = output_stream.stdout.read(width * height * 3)  # 3 Bytes pro Pixel (RGB)
                if not in_bytes:
                    break
                
                # Konvertiere Bytes zu NumPy-Array
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                frames.append(frame)
            
            # Beende den FFmpeg-Prozess
            output_stream.stdout.close()
            output_stream.wait()
            
            # Konvertiere Liste zu einem 4D-Array (Frames, Höhe, Breite, Kanäle)
            if frames:
                video_array = np.stack(frames)
                logger.info(f"{len(frames)} Frames aus {video_path} in NumPy-Array geladen")
                return video_array, metadata
            else:
                logger.warning(f"Keine Frames aus {video_path} geladen")
                return np.array([]), metadata
                
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg-Fehler beim Konvertieren von {video_path} zu NumPy: {e.stderr.decode() if e.stderr else str(e)}")
            return np.array([]), {}
        except Exception as e:
            logger.error(f"Fehler beim Konvertieren von {video_path} zu NumPy: {e}")
            return np.array([]), {}
    
    def process_directory(self, fps: float = 1.0) -> Dict[str, List[str]]:
        """
        Verarbeitet alle Videos im Eingabeverzeichnis.
        
        Args:
            fps: Frames pro Sekunde für die Extraktion
            
        Returns:
            Dictionary mit Video-Pfaden als Schlüssel und Listen von Frame-Pfaden als Werte
        """
        results = {}
        
        # Unterstützte Videoformate
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        
        # Finde alle Videos im Eingabeverzeichnis
        for video_file in self.input_dir.glob('**/*'):
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                try:
                    # Extrahiere Frames
                    extracted_frames = self.extract_frames(str(video_file), fps=fps)
                    
                    if extracted_frames:
                        results[str(video_file)] = extracted_frames
                        
                except Exception as e:
                    logger.error(f"Fehler bei der Verarbeitung von {video_file}: {e}")
        
        total_frames = sum(len(frames) for frames in results.values())
        logger.info(f"Insgesamt {total_frames} Frames aus {len(results)} Videos extrahiert")
        
        return results


if __name__ == "__main__":
    # Beispiel für die Verwendung des FrameExtractors
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
    input_dir = os.getenv('SCENE_OUTPUT_DIR', 'data/scenes')
    output_dir = os.getenv('FRAME_OUTPUT_DIR', 'data/frames')
    
    extractor = FrameExtractor(input_dir, output_dir)
    
    if len(sys.argv) > 1:
        # Verarbeite ein bestimmtes Video
        video_path = sys.argv[1]
        fps = 1.0  # 1 Frame pro Sekunde
        if len(sys.argv) > 2:
            fps = float(sys.argv[2])
        extractor.extract_frames(video_path, fps=fps)
    else:
        # Verarbeite alle Videos im Eingabeverzeichnis
        extractor.process_directory(fps=1.0) 