#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hauptmodul für die GUI des Video Batch Still Generator.
Startet die QML-Engine und lädt das Hauptfenster.
"""

import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot, Property, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from src.gui.controllers.main_controller import MainController


class GuiApplication:
    """Hauptklasse für die GUI-Anwendung."""
    
    def __init__(self):
        """Initialisiert die GUI-Anwendung."""
        # QT-Umgebungsvariablen für High-DPI-Displays
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
        
        # QML-Engine und Hauptfenster
        self.app = QGuiApplication(sys.argv)
        self.app.setOrganizationName("Video Batch Still Generator")
        self.app.setApplicationName("Video Batch Still Generator")
        
        self.engine = QQmlApplicationEngine()
        
        # Controller registrieren
        self.main_controller = MainController()
        
        # Controller als Kontext-Property für QML verfügbar machen
        self.engine.rootContext().setContextProperty("mainController", self.main_controller)
        
        # QML-Pfade
        qml_dir = Path(__file__).parent / "qml"
        self.engine.addImportPath(str(qml_dir))
        
        # Hauptfenster laden
        qml_file = qml_dir / "Main.qml"
        self.engine.load(QUrl.fromLocalFile(str(qml_file)))
        
        # Prüfen, ob das Laden erfolgreich war
        if not self.engine.rootObjects():
            sys.exit(-1)
    
    def run(self):
        """Startet die GUI-Anwendung."""
        return self.app.exec()


if __name__ == "__main__":
    app = GuiApplication()
    sys.exit(app.run()) 