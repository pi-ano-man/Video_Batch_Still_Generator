#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QualityEvaluator Modul - Bewertet Bildqualität und clustert ähnliche Bilder
"""

import os
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ImageQuality:
    """Datenklasse für Bildqualitätsmetriken."""
    path: str
    sharpness: float  # Laplace-Varianz für Schärfe
    motion_blur: float  # Bewegungsunschärfe-Score
    noise: float  # Rausch-Score
    exposure: float  # Belichtungswert (0 = ideal, negativ = unterbelichtet, positiv = überbelichtet)
    face_size: float  # Relative Größe des Gesichts im Bild (0-1)
    overall_score: float  # Gesamtbewertung
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Konvertiert die Bildqualitätsmetriken in ein Dictionary."""
        return {
            'path': self.path,
            'sharpness': self.sharpness,
            'motion_blur': self.motion_blur,
            'noise': self.noise,
            'exposure': self.exposure,
            'face_size': self.face_size,
            'overall_score': self.overall_score
        }


class QualityEvaluator:
    """
    Klasse zur Bewertung der Bildqualität und zum Clustering ähnlicher Bilder.
    Implementiert Laplace-Varianz für Schärfeerkennung und DBSCAN für Clustering.
    """
    
    def __init__(self, input_dir: str, 
                sharpness_weight: float = 0.5,
                motion_blur_weight: float = 0.3,
                exposure_weight: float = 0.1,
                face_size_weight: float = 0.1):
        """
        Initialisiert den QualityEvaluator.
        
        Args:
            input_dir: Verzeichnis mit den Kandidaten-Frames
            sharpness_weight: Gewichtung der Schärfe im Gesamtscore
            motion_blur_weight: Gewichtung der Bewegungsunschärfe im Gesamtscore
            exposure_weight: Gewichtung der Belichtung im Gesamtscore
            face_size_weight: Gewichtung der Gesichtsgröße im Gesamtscore
        """
        self.input_dir = Path(input_dir)
        
        # Gewichtungen für den Gesamtscore
        self.weights = {
            'sharpness': sharpness_weight,
            'motion_blur': motion_blur_weight,
            'exposure': exposure_weight,
            'face_size': face_size_weight
        }
        
        # Prüfe, ob die Summe der Gewichte 1 ist
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Summe der Gewichte ist {total_weight}, nicht 1.0. Normalisiere Gewichte.")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Feature-Extraktoren für ähnliches Bild-Clustering
        self.orb = cv2.ORB_create()
        
        logger.info(f"QualityEvaluator initialisiert mit input_dir={input_dir}")
    
    def compute_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Berechnet die Laplace-Varianz als Maß für die Bildschärfe.
        Höhere Werte bedeuten schärfere Bilder.
        
        Args:
            image: Das Eingabebild als NumPy-Array
            
        Returns:
            Laplace-Varianz (Schärfewert)
        """
        # Konvertiere zu Graustufen, falls es ein Farbbild ist
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Berechne Laplace-Varianz
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
    
    def detect_motion_blur(self, image: np.ndarray) -> float:
        """
        Erkennt Bewegungsunschärfe im Bild mit Hilfe der Fourier-Transformation.
        Niedrigere Werte deuten auf stärkere Bewegungsunschärfe hin.
        
        Args:
            image: Das Eingabebild als NumPy-Array
            
        Returns:
            Bewegungsunschärfe-Score (0-1, wobei 1 am besten ist)
        """
        # Konvertiere zu Graustufen, falls es ein Farbbild ist
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Berechne FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Analysiere Spektrum auf Bewegungsunschärfe
        # Bewegungsunschärfe erzeugt charakteristische Linien im Spektrum
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Extrahiere zentralen Bereich
        center_size = min(20, min(rows, cols) // 4)
        center = magnitude_spectrum[crow-center_size:crow+center_size, ccol-center_size:ccol+center_size]
        
        # Berechne statistische Merkmale des Spektrums
        std_dev = np.std(center)
        max_val = np.max(center)
        
        # Normalisiere zu einem Score zwischen 0-1
        # Höhere Standardabweichung und höhere Maximalwerte deuten auf weniger Bewegungsunschärfe hin
        motion_blur_score = min(1.0, (std_dev / 50.0) * (max_val / 200.0))
        
        return motion_blur_score
    
    def calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Schätzt den Rauschpegel im Bild. Höhere Werte bedeuten mehr Rauschen.
        
        Args:
            image: Das Eingabebild als NumPy-Array
            
        Returns:
            Rausch-Score
        """
        # Konvertiere zu Graustufen, falls es ein Farbbild ist
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Wende Gaußschen Weichzeichner an
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Berechne Differenz zwischen Original und geglättetem Bild
        noise = cv2.absdiff(gray, blur)
        
        # Berechne durchschnittliches Rauschen
        noise_level = np.mean(noise)
        
        return noise_level
    
    def analyze_exposure(self, image: np.ndarray) -> float:
        """
        Analysiert die Belichtung des Bildes.
        Werte nahe 0 sind ideal, negative Werte bedeuten Unterbelichtung,
        positive Werte bedeuten Überbelichtung.
        
        Args:
            image: Das Eingabebild als NumPy-Array
            
        Returns:
            Belichtungswert (-1 bis 1)
        """
        # Konvertiere zu Graustufen, falls es ein Farbbild ist
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Berechne Histogramm
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalisieren
        
        # Berechne gewichteten Durchschnitt
        weighted_avg = np.sum(hist * np.arange(256)) / 255.0
        
        # Mappe 0-1 Wertebereich zu -1 bis 1
        # 0.5 ist ideal, darunter ist unterbelichtet, darüber überbelichtet
        exposure = (weighted_avg - 0.5) * 2
        
        return exposure
    
    def estimate_face_size(self, image: np.ndarray, face_region: Optional[Dict] = None) -> float:
        """
        Schätzt die relative Größe eines Gesichts im Bild.
        Falls keine Gesichtskoordinaten übergeben werden, wird eine Gesichtserkennung durchgeführt.
        
        Args:
            image: Das Eingabebild als NumPy-Array
            face_region: Optional - Dictionary mit den Gesichtskoordinaten (x, y, w, h)
            
        Returns:
            Relative Gesichtsgröße (0-1)
        """
        if face_region is None:
            # Versuche, Gesichter mit OpenCV-Haarcascade zu erkennen
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Konvertiere zu Graustufen für die Gesichtserkennung
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                logger.debug("Kein Gesicht im Bild gefunden")
                return 0.0
            
            # Verwende das größte Gesicht
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
        else:
            # Verwende übergebene Gesichtskoordinaten
            x = face_region.get("x", 0)
            y = face_region.get("y", 0)
            w = face_region.get("w", 0)
            h = face_region.get("h", 0)
        
        # Berechne relative Größe (Anteil am Gesamtbild)
        image_area = image.shape[0] * image.shape[1]
        face_area = w * h
        
        if image_area == 0:
            return 0.0
            
        relative_size = face_area / image_area
        
        return relative_size
    
    def compute_quality_score(self, image_path: str, face_region: Optional[Dict] = None) -> ImageQuality:
        """
        Berechnet einen Qualitätsscore für ein Bild basierend auf verschiedenen Metriken.
        
        Args:
            image_path: Pfad zum Bild
            face_region: Optional - Dictionary mit den Gesichtskoordinaten
            
        Returns:
            ImageQuality-Objekt mit allen Metriken
        """
        try:
            # Lade Bild
            image = cv2.imread(image_path)
            
            if image is None:
                logger.error(f"Konnte {image_path} nicht lesen")
                # Rückgabe von Standardwerten im Fehlerfall
                return ImageQuality(
                    path=image_path,
                    sharpness=0.0,
                    motion_blur=0.0,
                    noise=0.0,
                    exposure=0.0,
                    face_size=0.0,
                    overall_score=0.0
                )
            
            # Berechne Qualitätsmetriken
            sharpness = self.compute_laplacian_variance(image)
            motion_blur = self.detect_motion_blur(image)
            noise = self.calculate_noise_level(image)
            exposure = self.analyze_exposure(image)
            face_size = self.estimate_face_size(image, face_region)
            
            # Normalisiere Werte
            # Schärfe: Höhere Werte sind besser, aber typisch im Bereich 0-1000
            normalized_sharpness = min(1.0, sharpness / 1000.0)
            
            # Rauschen: Niedrigere Werte sind besser, typisch im Bereich 0-30
            normalized_noise = max(0.0, 1.0 - (noise / 30.0))
            
            # Belichtung: Werte nahe 0 sind ideal (-1 bis 1)
            normalized_exposure = 1.0 - abs(exposure)
            
            # Gesichtsgröße: Größer ist oft besser, aber nicht zu groß
            # Ideal ist etwa 5-20% des Bildes
            if face_size < 0.05:
                normalized_face_size = face_size / 0.05  # Linear von 0 bis 0.05
            elif face_size <= 0.2:
                normalized_face_size = 1.0  # Optimal zwischen 0.05 und 0.2
            else:
                normalized_face_size = max(0.0, 1.0 - ((face_size - 0.2) / 0.3))  # Linear abfallend
            
            # Berechne gewichteten Gesamtscore
            overall_score = (
                self.weights['sharpness'] * normalized_sharpness +
                self.weights['motion_blur'] * motion_blur +
                self.weights['exposure'] * normalized_exposure +
                self.weights['face_size'] * normalized_face_size
            )
            
            # Erstelle ImageQuality-Objekt
            quality = ImageQuality(
                path=image_path,
                sharpness=sharpness,
                motion_blur=motion_blur,
                noise=noise,
                exposure=exposure,
                face_size=face_size,
                overall_score=overall_score
            )
            
            logger.debug(f"Qualitätsscore für {image_path}: {overall_score:.2f}")
            return quality
            
        except Exception as e:
            logger.error(f"Fehler bei der Qualitätsbewertung von {image_path}: {e}")
            # Rückgabe von Standardwerten im Fehlerfall
            return ImageQuality(
                path=image_path,
                sharpness=0.0,
                motion_blur=0.0,
                noise=0.0,
                exposure=0.0,
                face_size=0.0,
                overall_score=0.0
            )
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extrahiert Features aus einem Bild für das Clustering.
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Feature-Vektor
        """
        try:
            # Lade Bild
            image = cv2.imread(image_path)
            
            if image is None:
                logger.error(f"Konnte {image_path} nicht lesen")
                return np.array([])
            
            # Skaliere das Bild auf eine einheitliche Größe
            max_size = 500
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
            
            # Konvertiere zu Graustufen
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extrahiere ORB-Features
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(keypoints) < 10:
                # Nicht genug markante Punkte für ORB
                # Verwende einen histogrammbasierten Ansatz
                hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                return hist
            
            # Normalisiere die Deskriptoren
            descriptors_mean = np.mean(descriptors, axis=0)
            return descriptors_mean
            
        except Exception as e:
            logger.error(f"Fehler bei der Feature-Extraktion von {image_path}: {e}")
            return np.array([])
    
    def cluster_similar_images(self, image_paths: List[str], 
                             eps: float = 0.3, min_samples: int = 2) -> Dict[int, List[str]]:
        """
        Clustert ähnliche Bilder mit DBSCAN.
        
        Args:
            image_paths: Liste von Bildpfaden
            eps: DBSCAN-Parameter für maximale Nachbarschaftsdistanz
            min_samples: DBSCAN-Parameter für Mindestanzahl von Samples in einer Nachbarschaft
            
        Returns:
            Dictionary mit Cluster-IDs als Schlüssel und Listen von Bildpfaden als Werte
        """
        if not image_paths:
            logger.warning("Keine Bilder zum Clustern übergeben")
            return {}
        
        logger.info(f"Clustere {len(image_paths)} Bilder")
        
        # Extrahiere Features für jedes Bild
        features = []
        valid_paths = []
        
        for path in image_paths:
            feature = self.extract_features(path)
            if feature.size > 0:
                features.append(feature)
                valid_paths.append(path)
            else:
                logger.warning(f"Konnte keine Features aus {path} extrahieren")
        
        if not features:
            logger.warning("Keine Features extrahiert")
            return {}
        
        # Konvertiere Features zu NumPy-Array
        features_array = np.array(features)
        
        try:
            # Berechne Distanzmatrix
            distances = pairwise_distances(features_array, metric='euclidean')
            
            # Führe DBSCAN-Clustering durch
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = dbscan.fit_predict(distances)
            
            # Organisiere Bilder nach Clustern
            clusters = defaultdict(list)
            
            for i, label in enumerate(labels):
                # Label -1 bedeutet Rauschen (keine Cluster-Zuordnung)
                if label >= 0:
                    clusters[label].append(valid_paths[i])
                else:
                    # Bilder ohne Cluster bekommen eine eigene ID
                    clusters[f"single_{i}"] = [valid_paths[i]]
            
            logger.info(f"{len(clusters)} Cluster gefunden")
            
            # Konvertiere defaultdict zurück zu normalem dict
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Fehler beim Clustering: {e}")
            return {}
    
    def select_best_from_clusters(self, clusters: Dict[Any, List[str]], 
                                face_regions: Optional[Dict[str, Dict]] = None) -> List[str]:
        """
        Wählt das beste Bild aus jedem Cluster basierend auf der Qualitätsbewertung.
        
        Args:
            clusters: Dictionary mit Cluster-IDs und Listen von Bildpfaden
            face_regions: Optional - Dictionary mit Bildpfaden und Gesichtskoordinaten
            
        Returns:
            Liste der besten Bilder aus jedem Cluster
        """
        best_images = []
        
        for cluster_id, image_paths in clusters.items():
            if not image_paths:
                continue
                
            logger.debug(f"Wähle bestes Bild aus Cluster {cluster_id} mit {len(image_paths)} Bildern")
            
            # Bewerte die Qualität jedes Bildes im Cluster
            quality_scores = []
            
            for path in image_paths:
                # Verwende Gesichtskoordinaten, falls vorhanden
                face_region = face_regions.get(path) if face_regions else None
                quality = self.compute_quality_score(path, face_region)
                quality_scores.append((path, quality.overall_score))
            
            # Sortiere nach Qualitätsscore (absteigend)
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Wähle das Bild mit dem höchsten Score
            if quality_scores:
                best_path, score = quality_scores[0]
                best_images.append(best_path)
                logger.debug(f"Bestes Bild aus Cluster {cluster_id}: {best_path} mit Score {score:.2f}")
        
        logger.info(f"{len(best_images)} beste Bilder aus {len(clusters)} Clustern ausgewählt")
        return best_images
    
    def process_directory(self, target_person: Optional[str] = None, 
                        face_regions: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Verarbeitet alle Bilder im Eingabeverzeichnis.
        
        Args:
            target_person: Optional - Name der Zielperson
            face_regions: Optional - Dictionary mit Bildpfaden und Gesichtskoordinaten
            
        Returns:
            Dictionary mit Ergebnissen (Qualitätsscores, Cluster, beste Bilder)
        """
        # Bestimme das Verzeichnis basierend auf der Zielperson
        if target_person:
            base_dir = self.input_dir / target_person
        else:
            base_dir = self.input_dir
        
        # Prüfe, ob das Verzeichnis existiert
        if not base_dir.exists():
            logger.error(f"Verzeichnis {base_dir} existiert nicht")
            return {
                'quality_scores': {},
                'clusters': {},
                'best_images': []
            }
        
        # Sammle alle Bilder
        supported_formats = ('*.jpg', '*.jpeg', '*.png')
        image_paths = []
        
        for fmt in supported_formats:
            image_paths.extend([str(p) for p in base_dir.glob(fmt)])
        
        if not image_paths:
            logger.warning(f"Keine Bilder in {base_dir} gefunden")
            return {
                'quality_scores': {},
                'clusters': {},
                'best_images': []
            }
        
        logger.info(f"{len(image_paths)} Bilder in {base_dir} gefunden")
        
        # Berechne Qualitätsscores für alle Bilder
        quality_scores = {}
        
        for path in image_paths:
            face_region = face_regions.get(path) if face_regions else None
            quality = self.compute_quality_score(path, face_region)
            quality_scores[path] = quality.to_dict()
        
        # Clustere ähnliche Bilder
        clusters = self.cluster_similar_images(image_paths)
        
        # Wähle das beste Bild aus jedem Cluster
        best_images = self.select_best_from_clusters(clusters, face_regions)
        
        return {
            'quality_scores': quality_scores,
            'clusters': {str(k): v for k, v in clusters.items()},  # Konvertiere Cluster-IDs zu Strings
            'best_images': best_images
        }


if __name__ == "__main__":
    # Beispiel für die Verwendung des QualityEvaluators
    import sys
    import json
    from dotenv import load_dotenv
    
    # Lade Umgebungsvariablen
    load_dotenv()
    
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Eingabeverzeichnis aus .env oder Standardwert
    input_dir = os.getenv('CANDIDATE_DIR', 'data/candidates')
    
    evaluator = QualityEvaluator(input_dir)
    
    if len(sys.argv) > 1:
        # Verarbeite ein bestimmtes Bild oder Person
        target = sys.argv[1]
        
        if os.path.isfile(target):
            # Einzelnes Bild bewerten
            quality = evaluator.compute_quality_score(target)
            print(json.dumps(quality.to_dict(), indent=2))
        elif os.path.isdir(target):
            # Verzeichnis verarbeiten
            results = evaluator.process_directory()
            
            # Zeige die besten Bilder
            print(f"Beste Bilder ({len(results['best_images'])}):")
            for i, path in enumerate(results['best_images']):
                score = results['quality_scores'][path]['overall_score']
                print(f"{i+1}. {path} (Score: {score:.2f})")
                
        else:
            # Als Personenname interpretieren
            results = evaluator.process_directory(target_person=target)
            
            # Zeige die besten Bilder
            print(f"Beste Bilder für {target} ({len(results['best_images'])}):")
            for i, path in enumerate(results['best_images']):
                score = results['quality_scores'][path]['overall_score']
                print(f"{i+1}. {path} (Score: {score:.2f})")
    else:
        # Verarbeite das gesamte Eingabeverzeichnis
        results = evaluator.process_directory()
        
        # Zeige eine Zusammenfassung
        print(f"Verarbeitet: {len(results['quality_scores'])} Bilder")
        print(f"Cluster gefunden: {len(results['clusters'])}")
        print(f"Beste Bilder ausgewählt: {len(results['best_images'])}") 