#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unittest für die FaceFilter-Klasse.
"""

import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

import numpy as np
import cv2

# Import der zu testenden Klasse
from src.face_filter.face_filter import FaceFilter


class TestFaceFilter(unittest.TestCase):
    """Test-Suite für die FaceFilter-Klasse."""

    def setUp(self):
        """Vorbereitung für jeden Test."""
        # Erstelle temporäre Verzeichnisse für Tests
        self.temp_dir = tempfile.mkdtemp()
        self.reference_dir = os.path.join(self.temp_dir, "reference")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.test_images_dir = os.path.join(self.temp_dir, "test_images")
        
        # Erstelle die Verzeichnisse
        os.makedirs(self.reference_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.test_images_dir, exist_ok=True)
        
        # Erstelle Referenzdaten
        self.person1_dir = os.path.join(self.reference_dir, "person1")
        self.person2_dir = os.path.join(self.reference_dir, "person2")
        os.makedirs(self.person1_dir, exist_ok=True)
        os.makedirs(self.person2_dir, exist_ok=True)
        
        # Threshold für Gesichtserkennung
        self.face_match_threshold = 0.6
        
        # Erstelle einen FaceFilter mit den temporären Verzeichnissen
        self.filter = FaceFilter(
            self.reference_dir,
            self.output_dir,
            face_match_threshold=self.face_match_threshold
        )

    def tearDown(self):
        """Aufräumen nach jedem Test."""
        # Lösche temporäre Verzeichnisse
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Testet, ob der FaceFilter korrekt initialisiert wird."""
        self.assertEqual(self.filter.reference_dir, Path(self.reference_dir))
        self.assertEqual(self.filter.output_dir, Path(self.output_dir))
        self.assertEqual(self.filter.face_match_threshold, self.face_match_threshold)
        self.assertEqual(self.filter.model_name, 'VGG-Face')
        self.assertEqual(self.filter.detector_backend, 'opencv')
        self.assertEqual(self.filter.reference_embeddings, {})
        self.assertEqual(self.filter.reference_people, set())

    @patch('src.face_filter.face_filter.DeepFace.represent')
    def test_load_reference_embeddings(self, mock_represent):
        """Testet das Laden von Referenz-Embeddings."""
        # Erstelle Pseudo-Referenzbilder
        ref_image1 = os.path.join(self.person1_dir, "ref1.jpg")
        ref_image2 = os.path.join(self.person1_dir, "ref2.jpg")
        ref_image3 = os.path.join(self.person2_dir, "ref3.jpg")
        
        # Erstelle leere Dateien
        for img_path in [ref_image1, ref_image2, ref_image3]:
            with open(img_path, 'w') as f:
                f.write('')
        
        # Mock für DeepFace.represent
        # Erstelle verschiedene Embeddings für jedes Bild
        embeddings = [
            [{"embedding": np.array([0.1, 0.2, 0.3])}],  # für ref1.jpg
            [{"embedding": np.array([0.4, 0.5, 0.6])}],  # für ref2.jpg
            [{"embedding": np.array([0.7, 0.8, 0.9])}]   # für ref3.jpg
        ]
        
        mock_represent.side_effect = embeddings
        
        # Führe die zu testende Methode aus
        self.filter._load_reference_embeddings()
        
        # Überprüfungen
        self.assertEqual(len(self.filter.reference_people), 2)
        self.assertTrue("person1" in self.filter.reference_people)
        self.assertTrue("person2" in self.filter.reference_people)
        
        self.assertEqual(len(self.filter.reference_embeddings["person1"]), 2)
        self.assertEqual(len(self.filter.reference_embeddings["person2"]), 1)
        
        # Überprüfe, ob DeepFace.represent für alle Bilder aufgerufen wurde
        self.assertEqual(mock_represent.call_count, 3)

    @patch('src.face_filter.face_filter.DeepFace.extract_faces')
    def test_find_faces(self, mock_extract_faces):
        """Testet die Funktion zum Finden von Gesichtern in einem Bild."""
        # Erstelle ein Test-Bild
        test_image_path = os.path.join(self.test_images_dir, "test.jpg")
        with open(test_image_path, 'w') as f:
            f.write('')
        
        # Mock für DeepFace.extract_faces
        mock_faces = [
            {"facial_area": {"x": 10, "y": 20, "w": 100, "h": 100}},
            {"facial_area": {"x": 200, "y": 220, "w": 80, "h": 80}}
        ]
        mock_extract_faces.return_value = mock_faces
        
        # Führe die zu testende Methode aus
        result = self.filter.find_faces(test_image_path)
        
        # Überprüfungen
        self.assertEqual(result, mock_faces)
        mock_extract_faces.assert_called_once_with(
            img_path=test_image_path,
            detector_backend='opencv',
            enforce_detection=False
        )

    def test_compare_with_references_empty(self):
        """Testet den Vergleich mit leeren Referenzen."""
        # Ohne Referenz-Embeddings sollte eine leere Zeichenkette und unendlich zurückgegeben werden
        person, distance = self.filter.compare_with_references(np.array([0.1, 0.2, 0.3]))
        self.assertEqual(person, "")
        self.assertEqual(distance, float('inf'))

    @patch('src.face_filter.face_filter.FaceFilter._load_reference_embeddings')
    def test_compare_with_references(self, mock_load):
        """Testet den Vergleich mit Referenz-Embeddings."""
        # Manuelles Setzen der Referenz-Embeddings
        self.filter.reference_embeddings = {
            "person1": [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])],
            "person2": [np.array([0.7, 0.8, 0.9])]
        }
        self.filter.reference_people = {"person1", "person2"}
        
        # Test mit einem Embedding, das näher an person1 liegt
        person, distance = self.filter.compare_with_references(np.array([0.15, 0.25, 0.35]))
        self.assertEqual(person, "person1")
        self.assertLess(distance, 0.5)  # Sollte eine geringe Distanz sein
        
        # Test mit einem Embedding, das näher an person2 liegt
        person, distance = self.filter.compare_with_references(np.array([0.75, 0.85, 0.95]))
        self.assertEqual(person, "person2")
        self.assertLess(distance, 0.5)  # Sollte eine geringe Distanz sein
        
        # Test mit Zielparameter
        person, distance = self.filter.compare_with_references(
            np.array([0.75, 0.85, 0.95]), target_person="person1"
        )
        self.assertEqual(person, "person1")  # Sollte person1 sein, trotz größerer Distanz

    @patch('src.face_filter.face_filter.DeepFace.represent')
    @patch('src.face_filter.face_filter.FaceFilter.compare_with_references')
    def test_filter_frame(self, mock_compare, mock_represent):
        """Testet die Filterung eines einzelnen Frames."""
        # Erstelle ein Test-Bild
        test_image_path = os.path.join(self.test_images_dir, "test.jpg")
        with open(test_image_path, 'w') as f:
            f.write('')
        
        # Mock für DeepFace.represent
        mock_represent.return_value = [{
            "embedding": np.array([0.1, 0.2, 0.3]),
            "facial_area": {"x": 10, "y": 20, "w": 100, "h": 100}
        }]
        
        # Mock für compare_with_references
        mock_compare.return_value = ("person1", 0.2)  # Guter Match mit person1
        
        # Führe die zu testende Methode aus
        match_found, confidence, details = self.filter.filter_frame(test_image_path)
        
        # Überprüfungen
        self.assertTrue(match_found)
        self.assertAlmostEqual(confidence, 0.8, places=1)  # 1 - 0.2 = 0.8
        self.assertEqual(details["person"], "person1")
        self.assertEqual(details["distance"], 0.2)
        self.assertTrue(details["threshold_passed"])
        
        # Test mit einem schlechten Match
        mock_compare.return_value = ("person2", 0.9)  # Schlechter Match mit person2
        
        match_found, confidence, details = self.filter.filter_frame(test_image_path)
        
        # Überprüfungen
        self.assertFalse(match_found)
        self.assertAlmostEqual(confidence, 0.1, places=1)  # 1 - 0.9 = 0.1
        self.assertEqual(details["person"], "person2")
        self.assertEqual(details["distance"], 0.9)
        self.assertFalse(details["threshold_passed"])

    @patch('src.face_filter.face_filter.FaceFilter.filter_frame')
    @patch('src.face_filter.face_filter.cv2.imread')
    @patch('src.face_filter.face_filter.cv2.imwrite')
    def test_filter_frames(self, mock_imwrite, mock_imread, mock_filter_frame):
        """Testet die Filterung mehrerer Frames."""
        # Erstelle Test-Bilder
        test_image1 = os.path.join(self.test_images_dir, "test1.jpg")
        test_image2 = os.path.join(self.test_images_dir, "test2.jpg")
        test_image3 = os.path.join(self.test_images_dir, "test3.jpg")
        
        for img_path in [test_image1, test_image2, test_image3]:
            with open(img_path, 'w') as f:
                f.write('')
        
        # Mock für filter_frame
        mock_filter_frame.side_effect = [
            (True, 0.8, {"person": "person1", "distance": 0.2, "confidence": 0.8, "face_region": {"x": 10, "y": 20, "w": 100, "h": 100}, "threshold_passed": True}),
            (False, 0.1, {"person": "person2", "distance": 0.9, "confidence": 0.1, "face_region": {}, "threshold_passed": False}),
            (True, 0.7, {"person": "person1", "distance": 0.3, "confidence": 0.7, "face_region": {"x": 30, "y": 40, "w": 90, "h": 90}, "threshold_passed": True})
        ]
        
        # Mock für Bildoperationen
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Führe die zu testende Methode aus
        results = self.filter.filter_frames([test_image1, test_image2, test_image3], target_person="person1")
        
        # Überprüfungen
        self.assertEqual(len(results), 3)
        self.assertTrue(results[test_image1]["match"])
        self.assertFalse(results[test_image2]["match"])
        self.assertTrue(results[test_image3]["match"])
        
        # Überprüfe, ob die matchenden Bilder kopiert wurden
        self.assertEqual(mock_imread.call_count, 2)  # Nur die 2 matchenden Bilder
        self.assertEqual(mock_imwrite.call_count, 2)  # Nur die 2 matchenden Bilder

    @patch('src.face_filter.face_filter.DeepFace.represent')
    @patch('src.face_filter.face_filter.cv2.imread')
    @patch('src.face_filter.face_filter.cv2.imwrite')
    def test_add_reference_image(self, mock_imwrite, mock_imread, mock_represent):
        """Testet das Hinzufügen eines Referenzbildes."""
        # Erstelle ein Test-Bild
        test_image = os.path.join(self.test_images_dir, "ref_test.jpg")
        with open(test_image, 'w') as f:
            f.write('')
        
        # Mock für Bildoperationen
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Mock für DeepFace.represent
        mock_represent.return_value = [{
            "embedding": np.array([0.1, 0.2, 0.3])
        }]
        
        # Führe die zu testende Methode aus
        result = self.filter.add_reference_image(test_image, "new_person")
        
        # Überprüfungen
        self.assertTrue(result)
        self.assertTrue("new_person" in self.filter.reference_people)
        self.assertEqual(len(self.filter.reference_embeddings["new_person"]), 1)
        self.assertTrue(np.array_equal(self.filter.reference_embeddings["new_person"][0], np.array([0.1, 0.2, 0.3])))
        
        # Überprüfe, ob das Bild kopiert wurde
        mock_imread.assert_called_once_with(test_image)
        mock_imwrite.assert_called_once()


if __name__ == '__main__':
    unittest.main() 