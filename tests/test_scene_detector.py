#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unittest für die SceneDetector-Klasse.
"""

import os
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Import der zu testenden Klasse
from src.scene_detector.scene_detector import SceneDetector


class TestSceneDetector(unittest.TestCase):
    """Test-Suite für die SceneDetector-Klasse."""

    def setUp(self):
        """Vorbereitung für jeden Test."""
        # Erstelle temporäre Verzeichnisse für Tests
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Erstelle die Verzeichnisse
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Erstelle einen SceneDetector mit den temporären Verzeichnissen
        self.detector = SceneDetector(self.input_dir, self.output_dir)

    def tearDown(self):
        """Aufräumen nach jedem Test."""
        # Lösche temporäre Verzeichnisse
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Testet, ob der SceneDetector korrekt initialisiert wird."""
        self.assertEqual(self.detector.input_dir, Path(self.input_dir))
        self.assertEqual(self.detector.output_dir, Path(self.output_dir))

    @patch('src.scene_detector.scene_detector.detect')
    def test_detect_scenes(self, mock_detect):
        """Testet die Funktion zur Szenenerkennung."""
        # Vorbereitung der Mocks
        mock_scene_list = [MagicMock(), MagicMock()]  # Zwei Szenen
        mock_detect.return_value = mock_scene_list
        
        # Test mit Content-Detektor
        result = self.detector.detect_scenes("test_video.mp4", threshold=30.0, use_adaptive_detector=False)
        
        # Überprüfungen
        self.assertEqual(result, mock_scene_list)
        
        # Überprüfe, ob der richtige Detektor verwendet wurde
        mock_detect.assert_called_once()
        args, kwargs = mock_detect.call_args
        self.assertEqual(args[0], "test_video.mp4")
        # Der zweite Argument sollte eine ContentDetector-Instanz sein
        self.assertEqual(args[1].__class__.__name__, "ContentDetector")
        
        # Zurücksetzen und nochmal mit AdaptiveDetector testen
        mock_detect.reset_mock()
        mock_detect.return_value = mock_scene_list
        
        result = self.detector.detect_scenes("test_video.mp4", use_adaptive_detector=True)
        
        # Überprüfungen
        self.assertEqual(result, mock_scene_list)
        args, kwargs = mock_detect.call_args
        self.assertEqual(args[1].__class__.__name__, "AdaptiveDetector")

    @patch('src.scene_detector.scene_detector.video_splitter.split_video_ffmpeg')
    def test_split_video(self, mock_split):
        """Testet die Funktion zum Aufteilen eines Videos."""
        # Vorbereitung der Mocks
        mock_scene_list = [MagicMock(), MagicMock()]  # Zwei Szenen
        mock_output_files = ["scene1.mp4", "scene2.mp4"]
        mock_split.return_value = mock_output_files
        
        # Test mit Standard-Präfix
        result = self.detector.split_video("test_video.mp4", mock_scene_list)
        
        # Überprüfungen
        self.assertEqual(result, mock_output_files)
        
        # Überprüfe, ob split_video_ffmpeg korrekt aufgerufen wurde
        mock_split.assert_called_once()
        args, kwargs = mock_split.call_args
        self.assertEqual(args[0], "test_video.mp4")
        self.assertEqual(args[1], mock_scene_list)
        self.assertTrue("test_video" in str(args[2]))  # Präfix sollte der Videoname sein
        
        # Zurücksetzen und nochmal mit benutzerdefiniertem Präfix testen
        mock_split.reset_mock()
        mock_split.return_value = mock_output_files
        
        result = self.detector.split_video("test_video.mp4", mock_scene_list, output_prefix="custom_prefix")
        
        # Überprüfungen
        self.assertEqual(result, mock_output_files)
        args, kwargs = mock_split.call_args
        self.assertTrue("custom_prefix" in str(args[2]))

    @patch('src.scene_detector.scene_detector.SceneDetector.detect_scenes')
    @patch('src.scene_detector.scene_detector.SceneDetector.split_video')
    def test_process_video(self, mock_split, mock_detect):
        """Testet die Funktion zur Verarbeitung eines Videos."""
        # Vorbereitung der Mocks
        mock_scene_list = [MagicMock(), MagicMock()]  # Zwei Szenen
        mock_output_files = ["scene1.mp4", "scene2.mp4"]
        mock_detect.return_value = mock_scene_list
        mock_split.return_value = mock_output_files
        
        # Test
        result = self.detector.process_video("test_video.mp4", threshold=30.0, use_adaptive_detector=True)
        
        # Überprüfungen
        self.assertEqual(result, mock_output_files)
        
        # Überprüfe, ob die richtigen Funktionen aufgerufen wurden
        mock_detect.assert_called_once_with("test_video.mp4", 30.0, True)
        mock_split.assert_called_once_with("test_video.mp4", mock_scene_list)

    @patch('src.scene_detector.scene_detector.Path.glob')
    @patch('src.scene_detector.scene_detector.Path.is_file')
    @patch('src.scene_detector.scene_detector.SceneDetector.process_video')
    def test_process_directory(self, mock_process, mock_is_file, mock_glob):
        """Testet die Funktion zur Verarbeitung eines Verzeichnisses."""
        # Vorbereitung der Mocks
        mock_video_files = [
            Path(os.path.join(self.input_dir, "video1.mp4")),
            Path(os.path.join(self.input_dir, "video2.avi")),
            Path(os.path.join(self.input_dir, "video3.mov"))
        ]
        
        # Mock für glob, der eine Liste von Mock-Pfaden zurückgibt
        def mock_glob_side_effect(pattern):
            if "**/*.mp4" in str(pattern):
                return [mock_video_files[0]]
            elif "**/*.avi" in str(pattern):
                return [mock_video_files[1]]
            elif "**/*.mov" in str(pattern):
                return [mock_video_files[2]]
            elif "**/*.mkv" in str(pattern):
                return []
            elif "**/*" in str(pattern):
                return mock_video_files
            return []
        
        mock_glob.side_effect = mock_glob_side_effect
        
        # Mock für is_file, der immer True zurückgibt
        mock_is_file.return_value = True
        
        # Mock für process_video, der für jedes Video eine Liste von Szenendateien zurückgibt
        mock_process.side_effect = [
            ["video1_scene1.mp4", "video1_scene2.mp4"],
            ["video2_scene1.mp4"],
            ["video3_scene1.mp4", "video3_scene2.mp4", "video3_scene3.mp4"]
        ]
        
        # Test
        result = self.detector.process_directory(threshold=25.0, use_adaptive_detector=False)
        
        # Überprüfungen
        self.assertEqual(len(result), 6)  # Insgesamt 6 Szenen aus 3 Videos
        
        # Überprüfe, ob process_video für jedes Video aufgerufen wurde
        expected_calls = [
            call(str(mock_video_files[0]), 25.0, False),
            call(str(mock_video_files[1]), 25.0, False),
            call(str(mock_video_files[2]), 25.0, False)
        ]
        mock_process.assert_has_calls(expected_calls, any_order=True)


if __name__ == '__main__':
    unittest.main() 