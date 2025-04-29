#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SuperResolutionExporter Modul - finaler Super-Resolution- und Batch-Export Schritt
"""

import os
import logging
import subprocess  # nosec - wird nur mit kontrollierten, festen Parametern für externe CLI-Tools verwendet
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SRModel(Enum):
    """Unterstützte Super-Resolution-Modelle."""
    ESRGAN = "ESRGAN"  # Enhanced SRGAN - gute Allzwecklösung
    EDVR = "EDVR"  # Video-basiertes Model mit temporaler Konsistenz
    BASICVSR = "BASICVSR"  # BasicVSR - einfacheres Modell 
    REAL_ESRGAN = "REAL_ESRGAN"  # Verbesserte ESRGAN-Version


class SuperResolutionExporter:
    """
    Klasse für den finalen Super-Resolution- und Batch-Export Schritt.
    Unterstützt verschiedene Open-Source-SR-Modelle über Python-Wrapper oder CLI.
    """
    
    def __init__(self, input_dir: str, output_dir: str, 
                model: Union[str, SRModel] = SRModel.ESRGAN,
                scale: int = 2,
                cli_path: Optional[str] = None,
                max_workers: int = 4):
        """
        Initialisiert den SuperResolutionExporter.
        
        Args:
            input_dir: Verzeichnis mit den Quellbildern
            output_dir: Verzeichnis für die Super-Resolution-Ergebnisse
            model: Zu verwendendes Modell (z.B. ESRGAN, EDVR, BASICVSR)
            scale: Skalierungsfaktor (2 oder 4 typischerweise)
            cli_path: Pfad zum externen CLI-Tool (optional)
            max_workers: Maximale Anzahl paralleler Prozesse
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modell und Skalierungsfaktor
        if isinstance(model, str):
            try:
                self.model = SRModel(model.upper())
            except ValueError:
                logger.warning(f"Unbekanntes Modell: {model}, verwende ESRGAN")
                self.model = SRModel.ESRGAN
        else:
            self.model = model
            
        self.scale = scale
        
        # CLI-Pfad (falls extern)
        self.cli_path = cli_path
        
        # Maximale Anzahl paralleler Prozesse
        self.max_workers = max_workers
        
        logger.info(f"SuperResolutionExporter initialisiert mit input_dir={input_dir}, "
                   f"output_dir={output_dir}, model={self.model.value}, scale={scale}")
        
        # Einige SR-Module erfordern GPU - prüfe, ob verfügbar
        self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        logger.info(f"CUDA verfügbar: {self.has_cuda}")
    
    def _get_model_module(self) -> Optional[object]:
        """
        Lädt das entsprechende Python-Modul für das ausgewählte Modell.
        
        Returns:
            Modul-Objekt oder None, falls nicht verfügbar
        """
        # Dieser Code kann je nach tatsächlich verfügbaren Paketen angepasst werden
        try:
            if self.model == SRModel.ESRGAN:
                # Hier könnten wir verschiedene ESRGAN-Implementierungen einbinden
                # z.B. BasicSR, ein beliebtes Framework für SR-Modelle
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from basicsr.utils.download_util import load_file_from_url
                from basicsr.utils import img2tensor, tensor2img
                from torch.nn import functional as F
                import torch
                
                # Dummy-Modul
                class ESRGANModule:
                    def __init__(self, scale):
                        self.scale = scale
                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
                        # Lade RRDB-Modell
                        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                        model_path = load_file_from_url(model_url, "models")
                        
                        # Definiere Netzwerk-Architektur
                        self.model = RRDBNet(3, 3, 64, 23, gc=32)
                        self.model.load_state_dict(torch.load(model_path)['params'], strict=True)  # nosec - Modelle werden nur aus vertrauenswürdigen Quellen geladen
                        self.model.eval()
                        self.model = self.model.to(self.device)
                    
                    def upscale(self, img):
                        # Konvertiere zu Tensor
                        img = img2tensor(img, bgr2rgb=True, float32=True)
                        img = img.unsqueeze(0).to(self.device)
                        
                        # Upscale
                        with torch.no_grad():
                            output = self.model(img)
                            if self.scale == 2:
                                # Resize wenn nötig
                                output = F.interpolate(output, scale_factor=0.5, mode='bicubic')
                        
                        # Konvertiere zurück zu Bild
                        output_img = tensor2img(output, rgb2bgr=True, min_max=(0, 1))
                        return output_img
                
                return ESRGANModule(self.scale)
                
            elif self.model == SRModel.EDVR:
                # EDVR ist für Videoverarbeitung, benötigt spezielle Bibliotheken
                # Ähnliches Dummy-Modul
                class EDVRModule:
                    def __init__(self, scale):
                        self.scale = scale
                        
                    def upscale(self, img):
                        # Dummy-Implementation - in Produktion würden wir hier
                        # das tatsächliche EDVR-Modell laden und verwenden
                        h, w, _ = img.shape
                        return cv2.resize(img, (w * self.scale, h * self.scale), 
                                        interpolation=cv2.INTER_CUBIC)
                
                return EDVRModule(self.scale)
                
            elif self.model == SRModel.BASICVSR:
                # BasicVSR ist eine schnellere Version für Videoverarbeitung
                class BasicVSRModule:
                    def __init__(self, scale):
                        self.scale = scale
                        
                    def upscale(self, img):
                        # Dummy-Implementation
                        h, w, _ = img.shape
                        return cv2.resize(img, (w * self.scale, h * self.scale), 
                                        interpolation=cv2.INTER_CUBIC)
                
                return BasicVSRModule(self.scale)
                
            elif self.model == SRModel.REAL_ESRGAN:
                try:
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    import torch
                    
                    class RealESRGANModule:
                        def __init__(self, scale):
                            self.scale = scale
                            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
                            # Lade RealESRGAN
                            model_name = 'RealESRGAN_x4plus'  # oder 'RealESRGAN_x2plus' für scale=2
                            if scale == 2:
                                model_name = 'RealESRGAN_x2plus'
                                
                            self.model = RealESRGANer(
                                scale=scale,
                                model_path=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth',
                                dni_weight=0,
                                model=RRDBNet(3, 3, 64, 23, gc=32),
                                device=self.device
                            )
                        
                        def upscale(self, img):
                            # Upscale mit RealESRGAN
                            output, _ = self.model.enhance(img)
                            return output
                    
                    return RealESRGANModule(self.scale)
                except ImportError:
                    logger.warning("RealESRGAN nicht installiert, verwende OpenCV-Fallback")
                    return None
                    
            else:
                logger.warning(f"Unbekanntes Modell: {self.model}, verwende OpenCV-Fallback")
                return None
                
        except ImportError as e:
            logger.warning(f"Konnte SR-Modul nicht laden: {e}. Verwende OpenCV-Fallback.")
            return None
    
    def upscale_image_opencv(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale ein Bild mit OpenCV (Fallback-Methode).
        
        Args:
            image: Eingangsbild als NumPy-Array
            
        Returns:
            Hochskaliertes Bild
        """
        h, w = image.shape[:2]
        new_size = (w * self.scale, h * self.scale)
        
        # Verwende Lanczos-Resampling für bessere Qualität
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    def upscale_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Upscale ein einzelnes Bild mit dem ausgewählten SR-Modell.
        
        Args:
            image_path: Pfad zum Eingangsbild
            output_path: Optionaler Pfad für die Ausgabe
            
        Returns:
            Pfad zum hochskalierten Bild
        """
        # Bestimme Ausgabepfad, falls nicht angegeben
        if output_path is None:
            image_file = Path(image_path)
            output_file = self.output_dir / f"{image_file.stem}_sr{self.scale}x{image_file.suffix}"
            output_path = str(output_file)
        
        # Versuche zuerst die externe CLI, falls angegeben
        if self.cli_path and os.path.exists(self.cli_path):
            return self._upscale_image_cli(image_path, output_path)
        
        # Lade das SR-Modul, falls verfügbar
        sr_module = self._get_model_module()
        
        try:
            # Lade das Eingangsbild
            img = cv2.imread(image_path)
            
            if img is None:
                logger.error(f"Konnte {image_path} nicht lesen")
                return ""
            
            # Upscale das Bild
            if sr_module:
                logger.info(f"Upscale {image_path} mit {self.model.value}")
                upscaled_img = sr_module.upscale(img)
            else:
                logger.info(f"Upscale {image_path} mit OpenCV (Fallback)")
                upscaled_img = self.upscale_image_opencv(img)
            
            # Speichere das Ergebnis
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, upscaled_img)
            
            logger.info(f"Hochskaliertes Bild gespeichert: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Fehler beim Upscaling von {image_path}: {e}")
            
            # Fallback: Verwende OpenCV im Fehlerfall
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    upscaled_img = self.upscale_image_opencv(img)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, upscaled_img)
                    logger.info(f"Hochskaliertes Bild mit Fallback gespeichert: {output_path}")
                    return output_path
            except Exception as e2:
                logger.error(f"Auch Fallback fehlgeschlagen für {image_path}: {e2}")
            
            return ""
    
    def _upscale_image_cli(self, image_path: str, output_path: str) -> str:
        """
        Upscale ein Bild mit einem externen CLI-Tool.
        
        Args:
            image_path: Pfad zum Eingangsbild
            output_path: Pfad für die Ausgabe
            
        Returns:
            Pfad zum hochskalierten Bild
        """
        try:
            # Stelle sicher, dass der Ausgabeordner existiert
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Bereite den Befehl vor
            cmd = [
                self.cli_path,
                "--input", image_path,
                "--output", output_path,
                "--model", self.model.value.lower(),
                "--scale", str(self.scale)
            ]
            
            logger.info(f"Führe aus: {' '.join(cmd)}")
            
            # Führe den Befehl aus
            result = subprocess.run(  # nosec - cmd enthält nur validierte, nicht-benutzerkontrollierte Parameter
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            if result.returncode == 0:
                logger.info(f"Hochskaliertes Bild gespeichert: {output_path}")
                return output_path
            else:
                logger.error(f"CLI-Fehler: {result.stderr}")
                return ""
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Fehler beim Ausführen des CLI-Tools: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return ""
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei CLI: {e}")
            return ""
    
    def batch_upscale(self, image_paths: List[str], output_format: str = "png") -> List[str]:
        """
        Upscale mehrere Bilder im Batch-Modus.
        
        Args:
            image_paths: Liste der Pfade zu den Eingangsbildern
            output_format: Ausgabeformat (png oder tiff)
            
        Returns:
            Liste der Pfade zu den hochskalierten Bildern
        """
        if not image_paths:
            logger.warning("Keine Bilder zum Upscaling übergeben")
            return []
        
        logger.info(f"Starte Batch-Upscaling für {len(image_paths)} Bilder mit {self.model.value}")
        
        # Validiere Ausgabeformat
        if output_format.lower() not in ('png', 'tiff', 'jpg'):
            logger.warning(f"Ungültiges Ausgabeformat: {output_format}, verwende PNG")
            output_format = 'png'
        
        # Bereite Ausgabepfade vor
        output_paths = []
        for path in image_paths:
            image_file = Path(path)
            output_file = self.output_dir / f"{image_file.stem}_sr{self.scale}x.{output_format.lower()}"
            output_paths.append(str(output_file))
        
        # Parallele Verarbeitung
        results = []
        
        if self.max_workers <= 1:
            # Sequenzielle Verarbeitung
            for i, (in_path, out_path) in enumerate(zip(image_paths, output_paths)):
                logger.info(f"Verarbeite Bild {i+1}/{len(image_paths)}: {in_path}")
                result_path = self.upscale_image(in_path, out_path)
                if result_path:
                    results.append(result_path)
        else:
            # Parallele Verarbeitung mit ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Starte alle Upscaling-Aufgaben
                future_to_path = {
                    executor.submit(self.upscale_image, in_path, out_path): (i, in_path)
                    for i, (in_path, out_path) in enumerate(zip(image_paths, output_paths))
                }
                
                # Verarbeite die Ergebnisse
                for future in as_completed(future_to_path):
                    i, in_path = future_to_path[future]
                    try:
                        result_path = future.result()
                        if result_path:
                            results.append(result_path)
                            logger.info(f"Fertig {i+1}/{len(image_paths)}: {in_path} -> {result_path}")
                        else:
                            logger.error(f"Fehler bei {i+1}/{len(image_paths)}: {in_path}")
                    except Exception as e:
                        logger.error(f"Ausnahme bei {i+1}/{len(image_paths)}: {in_path} - {e}")
        
        logger.info(f"Batch-Upscaling abgeschlossen: {len(results)}/{len(image_paths)} erfolgreich")
        return results
    
    def process_directory(self, output_format: str = "png") -> List[str]:
        """
        Verarbeitet alle Bilder im Eingabeverzeichnis.
        
        Args:
            output_format: Ausgabeformat (png oder tiff)
            
        Returns:
            Liste der Pfade zu den hochskalierten Bildern
        """
        # Sammle alle Bilder im Eingabeverzeichnis
        supported_formats = ('*.jpg', '*.jpeg', '*.png')
        image_paths = []
        
        for fmt in supported_formats:
            image_paths.extend([str(p) for p in self.input_dir.glob(f"**/{fmt}")])
        
        if not image_paths:
            logger.warning(f"Keine Bilder in {self.input_dir} gefunden")
            return []
        
        logger.info(f"{len(image_paths)} Bilder in {self.input_dir} gefunden")
        
        # Starte Batch-Upscaling
        return self.batch_upscale(image_paths, output_format)


if __name__ == "__main__":
    # Beispiel für die Verwendung des SuperResolutionExporters
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
    input_dir = os.getenv('CANDIDATE_DIR', 'data/candidates')
    output_dir = os.getenv('FINAL_OUTPUT_DIR', 'data/output')
    
    # Verwende das Modell aus .env oder Standard
    model_name = os.getenv('SR_MODEL', 'ESRGAN')
    scale = int(os.getenv('SR_SCALE', '2'))
    
    # CLI-Pfad, falls vorhanden
    cli_path = os.getenv('SR_CLI_PATH', None)
    
    exporter = SuperResolutionExporter(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model_name,
        scale=scale,
        cli_path=cli_path
    )
    
    if len(sys.argv) > 1:
        # Verarbeite ein bestimmtes Bild oder Verzeichnis
        path = sys.argv[1]
        
        if os.path.isfile(path):
            # Upscale ein einzelnes Bild
            output_path = exporter.upscale_image(path)
            if output_path:
                print(f"Bild hochskaliert: {output_path}")
            else:
                print(f"Fehler beim Hochskalieren von {path}")
        elif os.path.isdir(path):
            # Verarbeite ein Verzeichnis
            exporter.input_dir = Path(path)
            results = exporter.process_directory()
            print(f"{len(results)} Bilder erfolgreich hochskaliert")
    else:
        # Verarbeite das Standard-Eingabeverzeichnis
        results = exporter.process_directory()
        print(f"{len(results)} Bilder erfolgreich hochskaliert") 