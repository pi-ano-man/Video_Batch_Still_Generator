#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FaceFilter Modul - Filtert Frames nach Gesichtserkennung mit Template-Ansatz
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set
import glob

import numpy as np
import cv2
from deepface import DeepFace

logger = logging.getLogger(__name__)


class FaceFilter:
    """
    Klasse zur Gesichtserkennung und Filterung von Frames.
    Verwendet einen Template-Ansatz mit mehreren Referenzbildern.
    """
    
    def __init__(self, reference_dir: str, output_dir: str, 
                face_match_threshold: float = 0.6):
        """
        Initialisiert den FaceFilter.
        
        Args:
            reference_dir: Verzeichnis mit Referenzbildern (pro Person ein Unterordner)
            output_dir: Verzeichnis für gefilterte Kandidaten-Frames
            face_match_threshold: Schwellenwert für die Gesichtserkennung
        """
        self.reference_dir = Path(reference_dir)
        self.output_dir = Path(output_dir)
        self.face_match_threshold = face_match_threshold
        
        # Modell für Gesichtserkennung
        self.model_name = 'VGG-Face'  # Alternativen: 'Facenet', 'OpenFace', 'DeepFace', 'ArcFace'
        self.detector_backend = 'opencv'  # Standard-Gesichtsdetektor
        
        # Lade Referenz-Embeddings
        self.reference_embeddings = {}
        self.reference_people = set()
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FaceFilter initialisiert mit reference_dir={reference_dir}, output_dir={output_dir} "
                   f"und face_match_threshold={face_match_threshold}")
        
        # Lade vorhandene Referenz-Bilder, wenn das Verzeichnis existiert
        if self.reference_dir.exists():
            self._load_reference_embeddings()
    
    def _load_reference_embeddings(self) -> None:
        """
        Lädt Referenz-Embeddings aus allen Personenordnern.
        Jeder Unterordner im Referenzverzeichnis entspricht einer Person.
        """
        # Leere die Referenz-Embeddings
        self.reference_embeddings = {}
        self.reference_people = set()
        
        # Durchsuche alle Unterordner im Referenzverzeichnis
        for person_dir in self.reference_dir.glob('*'):
            if person_dir.is_dir():
                person_name = person_dir.name
                logger.info(f"Lade Referenzbilder für Person: {person_name}")
                
                # Lade alle Bilder aus dem Personenordner
                person_embeddings = []
                supported_formats = ('*.jpg', '*.jpeg', '*.png')
                
                # Durchsuche nach Bildern mit unterstützten Formaten
                for fmt in supported_formats:
                    image_files = list(person_dir.glob(fmt))
                    
                    for img_file in image_files:
                        try:
                            # Verwende DeepFace, um das Embedding zu extrahieren
                            embedding_obj = DeepFace.represent(
                                img_path=str(img_file),
                                model_name=self.model_name,
                                detector_backend=self.detector_backend
                            )
                            
                            # DeepFace.represent gibt eine Liste von Dictionaries zurück,
                            # eines für jedes erkannte Gesicht
                            if embedding_obj and len(embedding_obj) > 0:
                                # Verwende das erste erkannte Gesicht
                                embedding = embedding_obj[0]["embedding"]
                                person_embeddings.append(embedding)
                                logger.debug(f"Embedding für {img_file} extrahiert")
                            else:
                                logger.warning(f"Kein Gesicht in {img_file} gefunden")
                        
                        except Exception as e:
                            logger.error(f"Fehler beim Extrahieren des Embeddings aus {img_file}: {e}")
                
                # Speichere die Embeddings der Person
                if person_embeddings:
                    self.reference_embeddings[person_name] = person_embeddings
                    self.reference_people.add(person_name)
                    logger.info(f"{len(person_embeddings)} Embeddings für {person_name} geladen")
                else:
                    logger.warning(f"Keine Embeddings für {person_name} gefunden")
        
        total_embeddings = sum(len(embs) for embs in self.reference_embeddings.values())
        logger.info(f"Insgesamt {total_embeddings} Referenz-Embeddings für {len(self.reference_people)} Personen geladen")
    
    def find_faces(self, image: Union[str, np.ndarray]) -> List[Dict]:
        """
        Findet Gesichter in einem Bild und gibt deren Koordinaten zurück.
        
        Args:
            image: Bildpfad oder NumPy-Array
            
        Returns:
            Liste mit gefundenen Gesichtern und ihren Koordinaten
        """
        try:
            # Analysiere Gesichter mit DeepFace
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=False  # False erlaubt keine Fehler, wenn kein Gesicht gefunden wird
            )
            
            logger.debug(f"{len(faces)} Gesichter im Bild gefunden")
            return faces
        
        except Exception as e:
            logger.error(f"Fehler beim Erkennen von Gesichtern: {e}")
            return []
    
    def compare_with_references(self, face_embedding: np.ndarray, 
                              target_person: Optional[str] = None) -> Tuple[str, float]:
        """
        Vergleicht ein Gesichts-Embedding mit allen Referenz-Embeddings.
        
        Args:
            face_embedding: Embedding des zu vergleichenden Gesichts
            target_person: Optionaler Name der Zielperson (wenn nur mit einer Person verglichen werden soll)
            
        Returns:
            Tuple aus Namen der ähnlichsten Person und Distanz
        """
        if not self.reference_embeddings:
            logger.warning("Keine Referenz-Embeddings vorhanden")
            return "", float('inf')
        
        best_match = ""
        min_distance = float('inf')
        
        # Personen zum Vergleich (entweder alle oder nur die Zielperson)
        people_to_compare = [target_person] if target_person else self.reference_people
        
        for person in people_to_compare:
            if person in self.reference_embeddings:
                # Vergleiche mit allen Referenz-Embeddings dieser Person
                for ref_embedding in self.reference_embeddings[person]:
                    # Berechne die euklidische Distanz
                    distance = np.linalg.norm(np.array(face_embedding) - np.array(ref_embedding))
                    
                    # Aktualisiere den besten Match
                    if distance < min_distance:
                        min_distance = distance
                        best_match = person
        
        return best_match, min_distance
    
    def filter_frame(self, frame_path: str, target_person: Optional[str] = None) -> Tuple[bool, float, Dict]:
        """
        Filtert ein Frame nach Gesichtserkennung.
        
        Args:
            frame_path: Pfad zum Frame
            target_person: Optionaler Name der Zielperson
            
        Returns:
            Tuple aus (Match gefunden, Konfidenz, Details)
        """
        try:
            # Extrahiere das Embedding des Frames
            embedding_obj = DeepFace.represent(
                img_path=frame_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if not embedding_obj or len(embedding_obj) == 0:
                logger.debug(f"Kein Gesicht in {frame_path} gefunden")
                return False, 0.0, {}
            
            # Verwende das erste gefundene Gesicht
            face_embedding = embedding_obj[0]["embedding"]
            face_region = embedding_obj[0].get("facial_area", {})
            
            # Vergleiche mit Referenz-Embeddings
            best_match, distance = self.compare_with_references(face_embedding, target_person)
            
            # Konvertiere Distanz in Konfidenz (0-1, wobei 1 ein perfekter Match ist)
            # Typische Schwellenwerte: 0.4-0.6 für gute Matches, abhängig vom Modell
            # Beachte, dass wir 1 - normalisierte_distanz verwenden
            max_distance = 1.0  # Maximale sinnvolle Distanz für Normalisierung
            confidence = max(0, 1 - (distance / max_distance))
            
            # Erstelle Details-Dictionary
            details = {
                "person": best_match,
                "distance": distance,
                "confidence": confidence,
                "face_region": face_region,
                "threshold_passed": confidence >= self.face_match_threshold
            }
            
            match_found = (confidence >= self.face_match_threshold)
            
            if match_found:
                logger.debug(f"Match für {best_match} in {frame_path} mit Konfidenz {confidence:.2f}")
            
            return match_found, confidence, details
            
        except Exception as e:
            logger.error(f"Fehler beim Filtern von {frame_path}: {e}")
            return False, 0.0, {"error": str(e)}
    
    def filter_frames(self, frames: List[str], target_person: Optional[str] = None, 
                     copy_matches: bool = True) -> Dict[str, Dict]:
        """
        Filtert eine Liste von Frames nach Gesichtserkennung.
        
        Args:
            frames: Liste von Frame-Pfaden
            target_person: Optionaler Name der Zielperson
            copy_matches: Ob übereinstimmende Frames kopiert werden sollen
            
        Returns:
            Dictionary mit Frame-Pfaden und Ergebnissen
        """
        results = {}
        matches = []
        
        logger.info(f"Filtere {len(frames)} Frames" + 
                   (f" für Person {target_person}" if target_person else ""))
        
        for i, frame_path in enumerate(frames):
            try:
                match_found, confidence, details = self.filter_frame(frame_path, target_person)
                
                results[frame_path] = {
                    "match": match_found,
                    "confidence": confidence,
                    **details
                }
                
                if match_found:
                    matches.append(frame_path)
                    
                    # Fortschrittsanzeige für große Datensätze
                    if i % 10 == 0 or i == len(frames) - 1:
                        logger.info(f"Fortschritt: {i+1}/{len(frames)} Frames verarbeitet, {len(matches)} Matches gefunden")
                
            except Exception as e:
                logger.error(f"Fehler bei Frame {frame_path}: {e}")
                results[frame_path] = {"match": False, "error": str(e)}
        
        logger.info(f"{len(matches)}/{len(frames)} Frames haben übereinstimmende Gesichter")
        
        # Kopiere übereinstimmende Frames in das Ausgabeverzeichnis
        if copy_matches and matches:
            person_dir = target_person if target_person else "matches"
            output_dir = self.output_dir / person_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Kopiere {len(matches)} übereinstimmende Frames nach {output_dir}")
            
            for frame_path in matches:
                try:
                    # Lese Quellbild
                    img = cv2.imread(frame_path)
                    
                    if img is None:
                        logger.error(f"Konnte {frame_path} nicht lesen")
                        continue
                    
                    # Erstelle Ausgabepfad
                    output_file = output_dir / Path(frame_path).name
                    
                    # Optional: Zeichne Gesichtsrahmen und Konfidenz
                    if results[frame_path].get("face_region"):
                        region = results[frame_path]["face_region"]
                        confidence = results[frame_path]["confidence"]
                        
                        # Zeichne Rahmen
                        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Füge Konfidenzwert hinzu
                        text = f"{confidence:.2f}"
                        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Speichere Bild
                    cv2.imwrite(str(output_file), img)
                    
                except Exception as e:
                    logger.error(f"Fehler beim Kopieren von {frame_path}: {e}")
        
        return results
    
    def add_reference_image(self, image_path: str, person_name: str) -> bool:
        """
        Fügt ein neues Referenzbild für eine Person hinzu.
        
        Args:
            image_path: Pfad zum Referenzbild
            person_name: Name der Person
            
        Returns:
            True, wenn das Bild erfolgreich hinzugefügt wurde
        """
        try:
            # Erstelle Personenverzeichnis, falls es nicht existiert
            person_dir = self.reference_dir / person_name
            person_dir.mkdir(parents=True, exist_ok=True)
            
            # Bestimme Ziel-Dateipfad
            image_file = Path(image_path)
            target_path = person_dir / image_file.name
            
            # Kopiere Bild ins Referenzverzeichnis
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Konnte {image_path} nicht lesen")
                return False
            
            cv2.imwrite(str(target_path), img)
            
            # Extrahiere Embedding und füge es hinzu
            embedding_obj = DeepFace.represent(
                img_path=str(target_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend
            )
            
            if embedding_obj and len(embedding_obj) > 0:
                embedding = embedding_obj[0]["embedding"]
                
                # Initialisiere Liste für Person, falls sie noch nicht existiert
                if person_name not in self.reference_embeddings:
                    self.reference_embeddings[person_name] = []
                    self.reference_people.add(person_name)
                
                # Füge Embedding hinzu
                self.reference_embeddings[person_name].append(embedding)
                
                logger.info(f"Referenzbild {image_file.name} für {person_name} hinzugefügt")
                return True
            else:
                logger.warning(f"Kein Gesicht in {image_path} gefunden")
                return False
                
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von {image_path} als Referenzbild: {e}")
            return False
    
    def get_reference_people(self) -> Set[str]:
        """
        Gibt die Namen aller Personen zurück, für die Referenzbilder vorhanden sind.
        
        Returns:
            Set mit den Namen der Personen
        """
        return self.reference_people


if __name__ == "__main__":
    # Beispiel für die Verwendung des FaceFilters
    import sys
    from dotenv import load_dotenv
    
    # Lade Umgebungsvariablen
    load_dotenv()
    
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verzeichnisse aus .env oder Standardwerte
    reference_dir = os.getenv('REFERENCE_IMAGES_DIR', 'data/reference')
    frame_dir = os.getenv('FRAME_OUTPUT_DIR', 'data/frames')
    output_dir = os.getenv('CANDIDATE_DIR', 'data/candidates')
    
    # Schwellenwert aus .env oder Standardwert
    threshold = float(os.getenv('FACE_MATCH_THRESHOLD', '0.6'))
    
    face_filter = FaceFilter(reference_dir, output_dir, face_match_threshold=threshold)
    
    if len(sys.argv) > 1:
        # Verarbeite einen bestimmten Frame oder ein Verzeichnis
        path = sys.argv[1]
        
        if os.path.isdir(path):
            # Alle Bilder im Verzeichnis verarbeiten
            supported_formats = ('*.jpg', '*.jpeg', '*.png')
            frames = []
            
            for fmt in supported_formats:
                frames.extend(glob.glob(os.path.join(path, fmt)))
            
            # Optionales zweites Argument für die Zielperson
            target_person = sys.argv[2] if len(sys.argv) > 2 else None
            
            face_filter.filter_frames(frames, target_person=target_person)
        else:
            # Einzelnes Bild verarbeiten
            match_found, confidence, details = face_filter.filter_frame(path)
            print(f"Match gefunden: {match_found}, Konfidenz: {confidence:.2f}")
            print(f"Details: {details}")
    else:
        # Liste alle registrierten Personen auf
        people = face_filter.get_reference_people()
        print(f"Registrierte Personen: {', '.join(people) if people else 'Keine'}")
        
        # Beispiel für die Filterung aller Frames
        print("Verwende den Befehl: python face_filter.py <frame_dir_oder_bild> [person_name]") 