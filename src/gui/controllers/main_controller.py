#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Haupt-Controller für das GUI.
Verwaltet die Verbindung zwischen QML-Frontend und Backend-Logik.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from shutil import copy2
import subprocess
import tempfile
import time

from PySide6.QtCore import QObject, Signal, Slot, Property, QUrl, QStringListModel

# Importiere die Hauptklasse aus dem src-Modul
try:
    from src.main import BatchStillGenerator
    print("Echter BatchStillGenerator erfolgreich importiert")
except ImportError as e:
    print(f"WARNUNG: Echter BatchStillGenerator konnte nicht importiert werden: {e}")
    print("Verwende Dummy-Implementierung für die Demo")
    
    # Dummy-Klasse für die Demo
    class BatchStillGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            print("Dummy BatchStillGenerator initialisiert mit:", kwargs)
        
        def process_directory(self, **kwargs):
            print("Dummy process_directory aufgerufen mit:", kwargs)
            return {"status": "Simulation: Verarbeitung läuft"}
            
        def process_video(self, video_path, target_person=None, fps=1.0, output_format="png"):
            print(f"Dummy process_video aufgerufen für: {video_path}, Person: {target_person}")
            return {"status": "Simulation: Video verarbeitet", "video_path": video_path}


class MainController(QObject):
    """Haupt-Controller für die GUI-Anwendung."""
    
    # Signale für die QML-Kommunikation
    videoListChanged = Signal()
    selectedImagesChanged = Signal()
    processingStateChanged = Signal()
    errorOccurred = Signal(str)
    
    def __init__(self, parent=None):
        """Initialisiert den Haupt-Controller."""
        super().__init__(parent)
        
        # Dateipfade für Daten
        self.video_input_dir = str(Path("data/raw_videos").absolute())
        self.scene_output_dir = str(Path("data/scenes").absolute())
        self.frame_output_dir = str(Path("data/frames").absolute())
        self.candidate_dir = str(Path("data/candidates").absolute())
        self.reference_dir = str(Path("data/reference").absolute())
        self.final_output_dir = str(Path("data/output").absolute())
        
        # Erstelle Verzeichnisse, falls nicht vorhanden
        for directory in [
            self.video_input_dir, 
            self.scene_output_dir, 
            self.frame_output_dir,
            self.candidate_dir,
            self.reference_dir,
            self.final_output_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Zustands-Properties
        self._videos = []
        self._selected_images = []
        self._processing = False
        self._current_person = ""
        
        # Initialisiere den Generator
        self._generator = BatchStillGenerator(
            video_input_dir=self.video_input_dir,
            scene_output_dir=self.scene_output_dir,
            frame_output_dir=self.frame_output_dir,
            candidate_dir=self.candidate_dir,
            reference_dir=self.reference_dir,
            final_output_dir=self.final_output_dir,
            face_match_threshold=0.6,
            sr_model="ESRGAN",
            sr_scale=2
        )
        
        # Lade verfügbare Videos
        self._load_videos()
    
    # --- Property: videos ---
    def _get_videos(self) -> List[str]:
        return self._videos
    
    @Slot(str, result=str)
    def get_video_thumbnail(self, video_path: str) -> str:
        """Generiert ein Thumbnail für ein Video und gibt den Pfad zurück."""
        try:
            # Erstelle Cache-Verzeichnis, falls nicht vorhanden
            cache_dir = Path("data/thumbnails").absolute()
            os.makedirs(cache_dir, exist_ok=True)
            
            # Erzeuge eindeutigen Dateinamen für das Thumbnail
            video_file = Path(video_path)
            thumbnail_name = f"{video_file.stem}_thumb.jpg"
            thumbnail_path = cache_dir / thumbnail_name
            
            # Prüfe, ob Thumbnail bereits existiert
            if thumbnail_path.exists():
                print(f"Thumbnail existiert bereits: {thumbnail_path}")
                return str(thumbnail_path.absolute())
            
            # Generiere Thumbnail mit FFmpeg (erster Frame)
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", "thumbnail,scale=160:90:force_original_aspect_ratio=decrease,pad=160:90:(ow-iw)/2:(oh-ih)/2",
                "-frames:v", "1",
                str(thumbnail_path.absolute()),
                "-y"  # Überschreibe vorhandene Datei
            ]
            
            print(f"Generiere Thumbnail für {video_path}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                print(f"Fehler beim Generieren des Thumbnails: {process.stderr}")
                return ""
            
            print(f"Thumbnail erstellt: {thumbnail_path}")
            return str(thumbnail_path.absolute())
            
        except Exception as e:
            print(f"Fehler beim Generieren des Thumbnails: {str(e)}")
            return ""
    
    @Slot()
    def _load_videos(self) -> None:
        """Lädt die verfügbaren Videos aus dem Eingabeverzeichnis."""
        video_dir = Path(self.video_input_dir)
        self._videos = []
        
        print(f"Lade Videos aus {video_dir}")
        
        if video_dir.exists():
            # Case-insensitive Erweiterungen
            video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg']
            
            # Zunächst alle Dateien im Verzeichnis sammeln
            all_files = []
            for file in video_dir.glob('*'):
                if file.is_file():
                    all_files.append(file)
            
            # Jede Datei prüfen und zur Liste hinzufügen, wenn es ein Video ist
            for file in all_files:
                suffix = file.suffix.lower()
                if suffix in video_exts:
                    self._videos.append(str(file))
                    print(f"Video gefunden: {file.name}")
                else:
                    print(f"Nicht-Video übersprungen: {file.name} (Endung: {suffix})")
            
            # Sortiere Videos alphabetisch
            self._videos.sort()
            
            print(f"Insgesamt {len(self._videos)} Videos geladen")
        else:
            print(f"Verzeichnis {video_dir} existiert nicht")
        
        self.videoListChanged.emit()
    
    videos = Property(list, _get_videos, notify=videoListChanged)
    
    # --- Property: selectedImages ---
    def _get_selected_images(self) -> List[str]:
        return self._selected_images
    
    selected_images = Property(list, _get_selected_images, notify=selectedImagesChanged)
    
    # --- Property: processing ---
    def _get_processing(self) -> bool:
        return self._processing
    
    def _set_processing(self, value: bool) -> None:
        if self._processing != value:
            self._processing = value
            self.processingStateChanged.emit()
    
    processing = Property(bool, _get_processing, _set_processing, notify=processingStateChanged)
    
    # --- Slots für QML-Interaktion ---
    @Slot(str)
    def set_current_person(self, person: str) -> None:
        """Setzt die aktuelle Zielperson für die Gesichtserkennung."""
        self._current_person = person
    
    @Slot(result=str)
    def get_current_person(self) -> str:
        """Gibt die aktuell ausgewählte Zielperson zurück."""
        return self._current_person
    
    @Slot(str, result=list)
    def get_reference_images(self, person: str) -> List[str]:
        """Gibt die Referenzbilder für eine Person zurück."""
        reference_path = Path(self.reference_dir) / person
        images = []
        
        if reference_path.exists():
            for ext in ['.jpg', '.jpeg', '.png']:
                images.extend([str(f) for f in reference_path.glob(f'*{ext}')])
        
        return images
    
    @Slot(str, result=bool)
    def process_directory(self, target_person: str) -> bool:
        """Verarbeitet alle Videos im Eingabeverzeichnis."""
        try:
            self._set_processing(True)
            
            # In der vollständigen Implementierung würde dies asynchron erfolgen
            result = self._generator.process_directory(
                target_person=target_person,
                fps=1.0,
                output_format="png"
            )
            
            print(f"Verarbeitungsergebnis: {result}")
            
            self._set_processing(False)
            return True
        except Exception as e:
            self.errorOccurred.emit(f"Fehler bei der Verarbeitung: {str(e)}")
            self._set_processing(False)
            return False
    
    @Slot(str, result=bool)
    def process_video(self, video_path: str) -> bool:
        """Verarbeitet ein einzelnes Video für die ausgewählte Person."""
        try:
            if not self._current_person:
                # Keine Person ausgewählt
                self.errorOccurred.emit("Keine Person ausgewählt. Bitte wählen Sie eine Person aus.")
                return False
                
            print(f"Verarbeite Video: {video_path} für Person: {self._current_person}")
            self._set_processing(True)
            
            try:
                # Echte Verarbeitung mit dem Generator
                result = self._generator.process_video(
                    video_path=video_path,
                    target_person=self._current_person,
                    fps=1.0,
                    output_format="png"
                )
                
                print(f"Verarbeitungsergebnis: {result}")
                self._set_processing(False)
                return True
            except Exception as e:
                print(f"Fehler bei der Verarbeitung des Videos: {str(e)}")
                self.errorOccurred.emit(f"Fehler bei der Verarbeitung: {str(e)}")
                self._set_processing(False)
                return False
                
        except Exception as e:
            print(f"Fehler bei der Verarbeitung des Videos: {str(e)}")
            self.errorOccurred.emit(f"Fehler bei der Verarbeitung: {str(e)}")
            self._set_processing(False)
            return False
    
    @Slot(list)
    def set_selected_images(self, images: List[str]) -> None:
        """Speichert die ausgewählten Bilder."""
        self._selected_images = images
        self.selectedImagesChanged.emit()
    
    @Slot(list, result=bool)
    def export_images(self, images: List[str]) -> bool:
        """Exportiert die ausgewählten Bilder mit Super-Resolution."""
        try:
            self._set_processing(True)
            
            # TODO: Echte Implementierung mit Fortschrittsanzeige
            print(f"Würde {len(images)} Bilder exportieren")
            
            self._set_processing(False)
            return True
        except Exception as e:
            self.errorOccurred.emit(f"Fehler beim Export: {str(e)}")
            self._set_processing(False)
            return False
    
    @Slot(result=list)
    def get_people(self) -> List[str]:
        """Gibt die Liste der Personen zurück, für die Referenzbilder existieren."""
        reference_path = Path(self.reference_dir)
        
        if not reference_path.exists():
            return []
        
        # Finde alle Unterordner im Referenzverzeichnis (je ein Ordner pro Person)
        return [f.name for f in reference_path.glob('*') if f.is_dir()]
    
    @Slot()
    def refresh_data(self) -> None:
        """Aktualisiert alle Daten."""
        self._load_videos()
    
    @Slot(list)
    def add_files(self, paths: List[str]) -> None:
        """Fügt Videos/Ordner hinzu, kopiert alle verarbeitbaren Videos ins Zielverzeichnis und aktualisiert die Liste."""
        try:
            video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg']
            added = 0
            skipped = 0
            not_video = 0
            
            print(f"Versuche, {len(paths)} Pfade zu verarbeiten...")
            
            for path in paths:
                print(f"Verarbeite Pfad: {path}")
                p = Path(path)
                
                if p.is_file():
                    # Prüfe Case-insensitive, ob die Dateiendung unterstützt wird
                    suffix = p.suffix.lower()
                    if suffix in video_exts:
                        dest = Path(self.video_input_dir) / p.name
                        print(f"Kopiere Video: {p} -> {dest}")
                        copy2(p, dest)
                        added += 1
                    else:
                        print(f"Überspringe Nicht-Video-Datei: {p} (Endung: {suffix})")
                        not_video += 1
                
                elif p.is_dir():
                    print(f"Durchsuche Verzeichnis: {p}")
                    video_count = 0
                    
                    for file in p.glob('**/*'):  # Benutzt glob statt rglob für bessere Kontrolle
                        if file.is_file():
                            suffix = file.suffix.lower()
                            if suffix in video_exts:
                                dest = Path(self.video_input_dir) / file.name
                                if dest.exists():
                                    print(f"Überspringe bereits existierende Datei: {file.name}")
                                    skipped += 1
                                else:
                                    print(f"Kopiere Video aus Ordner: {file} -> {dest}")
                                    copy2(file, dest)
                                    added += 1
                                    video_count += 1
                            else:
                                not_video += 1
                    
                    print(f"Gefundene Videos im Verzeichnis {p}: {video_count}")
            
            # Log-Ausgabe für Debugging
            print(f"Zusammenfassung: {added} Videos hinzugefügt, {skipped} übersprungen, {not_video} Nicht-Video-Dateien")
            
            self._load_videos()
            
            if added == 0:
                if not_video > 0:
                    self.errorOccurred.emit(f"Keine verarbeitbaren Videos gefunden. {not_video} Dateien waren keine Videos.")
                else:
                    self.errorOccurred.emit("Keine verarbeitbaren Videos gefunden.")
        except Exception as e:
            import traceback
            print(f"Fehler beim Hinzufügen: {str(e)}")
            print(traceback.format_exc())
            self.errorOccurred.emit(f"Fehler beim Hinzufügen: {str(e)}")
    
    @Slot(str, result=bool)
    def remove_video(self, video_path: str) -> bool:
        """Entfernt ein Video aus dem Eingabeverzeichnis."""
        try:
            video_file = Path(video_path)
            if video_file.exists():
                # Lösche das Video
                os.remove(video_file)
                print(f"Video gelöscht: {video_file}")
                
                # Lösche auch den Thumbnail, falls vorhanden
                thumb_name = f"{video_file.stem}_thumb.jpg"
                thumb_path = Path("data/thumbnails") / thumb_name
                if thumb_path.exists():
                    os.remove(thumb_path)
                    print(f"Thumbnail gelöscht: {thumb_path}")
                
                # Aktualisiere die Liste
                self._load_videos()
                return True
            return False
        except Exception as e:
            print(f"Fehler beim Entfernen des Videos: {str(e)}")
            self.errorOccurred.emit(f"Fehler beim Entfernen: {str(e)}")
            return False
    
    @Slot(result=bool)
    def remove_all_videos(self) -> bool:
        """Entfernt alle Videos aus dem Eingabeverzeichnis."""
        try:
            video_dir = Path(self.video_input_dir)
            video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg']
            thumb_dir = Path("data/thumbnails")
            
            # Lösche alle Videos
            count = 0
            for file in video_dir.glob('*'):
                if file.is_file() and file.suffix.lower() in video_exts:
                    os.remove(file)
                    print(f"Video gelöscht: {file}")
                    count += 1
                    
                    # Lösche auch den Thumbnail
                    thumb_name = f"{file.stem}_thumb.jpg"
                    thumb_path = thumb_dir / thumb_name
                    if thumb_path.exists():
                        os.remove(thumb_path)
            
            print(f"{count} Videos gelöscht")
            
            # Aktualisiere die Liste
            self._load_videos()
            return True
        except Exception as e:
            print(f"Fehler beim Entfernen aller Videos: {str(e)}")
            self.errorOccurred.emit(f"Fehler beim Entfernen aller Videos: {str(e)}")
            return False 