import os
import shutil
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from tqdm import tqdm
import sqlite3
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple

from dataclasses import dataclass
from typing import Optional, List, Tuple

# === Data Classes for Simplified Function Signatures ===

@dataclass
class FileInfo:
    """Container for file processing information."""
    relative_path: str
    absolute_path: str
    size: int
    mtime: float
    width: int = 0
    height: int = 0

@dataclass
class DetectionResult:
    """Container for YOLO detection results."""
    boxes: Optional[object] = None
    labels: Optional[object] = None
    confidences: Optional[object] = None
    has_error: bool = False
    discard_reasons: List[str] = None
    needs_manual_decision: bool = False
    
    def __post_init__(self):
        if self.discard_reasons is None:
            self.discard_reasons = []

@dataclass 
class ProcessingStats:
    """Container for processing statistics."""
    total_files: int = 0
    processed_files: int = 0
    cached_files: int = 0
    auto_sorted: int = 0
    manual_decisions: int = 0
    manual_sorted: int = 0
    kept_files: int = 0
    errors: int = 0

@dataclass
class ProcessingContext:
    """Container for processing context and configuration."""
    source_folder: str
    target_folder: str
    db_path: str
    stats: ProcessingStats
    pbar: object

# === Konfiguration ===
CONFIG = {
    'source_folder': "~/wallpaper/gallery-dl/",
    'target_folder': "~/wallpaper/discarded/",  # Ordner für unerwünschte Bilder
    'model_path': "yolov8s.pt",  # vortrainiertes YOLOv8-Modell
    'device': 'cpu',
    'target_classes': ['car', 'person', 'motorcycle'],  # Zielklassen: Autos und erkennbare Personen
    'large_area_threshold': 200000,  # Mindestfläche für große Objekte (≥200k + hohe Confidence = aussortieren)
    'small_area_threshold': 50000,  # Maximale Fläche für kleine Objekte (≤75k = automatisch behalten)
    'min_confidence_threshold': 0.25,  # Mindest-Confidence-Score (darunter wird ignoriert)
    'max_confidence_threshold': 0.7,  # Maximaler Confidence-Schwelle (≥0.7 bei großen Objekten = aussortieren)
    'min_resolution': [1920, 1080],  # Mindestauflösung [Breite, Höhe] für Wallpaper
    'aspect_ratio_tolerance': 0.1,  # Toleranz für Seitenverhältnis-Abweichung
    'supported_extensions': (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
}

# Extrahiere Konfigurationswerte für Kompatibilität
source_folder = os.path.expanduser(CONFIG['source_folder'])
target_folder = os.path.expanduser(CONFIG['target_folder'])
model_path = CONFIG['model_path']
device = CONFIG['device']
target_classes = CONFIG['target_classes']
large_area_threshold = CONFIG['large_area_threshold']
small_area_threshold = CONFIG['small_area_threshold']
min_confidence_threshold = CONFIG['min_confidence_threshold']
max_confidence_threshold = CONFIG['max_confidence_threshold']
min_resolution = CONFIG['min_resolution']
target_aspect_ratio = min_resolution[0] / min_resolution[1]  # Berechne aus der Mindestauflösung
aspect_ratio_tolerance = CONFIG['aspect_ratio_tolerance']

# === Modell laden ===
model = YOLO(model_path)

# === Database Functions ===
def get_config_hash():
    """
    Erstellt einen MD5-Hash der aktuellen Konfiguration für Cache-Invalidierung.
    
    Returns:
        str: MD5-Hash der serialisierten CONFIG als Hexadezimal-String
    """
    config_str = json.dumps(CONFIG, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def init_database():
    """
    Initialisiert die SQLite-Datenbank im Source-Ordner mit den erforderlichen Tabellen.
    
    Erstellt Tabellen für Konfiguration und verarbeitete Dateien. Prüft ob die aktuelle
    Konfiguration bereits existiert und setzt die Datenbank zurück falls sich die
    Konfiguration geändert hat.
    
    Returns:
        str: Absoluter Pfad zur Datenbankdatei
    
    Raises:
        Rekursiver Aufruf bei Datenbankfehlern nach Bereinigung
    """
    db_path = os.path.join(source_folder, '.image_filter_cache.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Tabelle für Konfiguration
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY,
                config_hash TEXT UNIQUE,
                config_data TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Tabelle für verarbeitete Dateien
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                config_hash TEXT,
                file_size INTEGER,
                file_mtime REAL,
                decision TEXT,
                discard_reasons TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (config_hash) REFERENCES config (config_hash)
            )
        ''')
        
        conn.commit()
        
        # Prüfe ob aktuelle Konfiguration existiert
        current_hash = get_config_hash()
        cursor.execute('SELECT config_hash FROM config WHERE config_hash = ?', (current_hash,))
        
        if not cursor.fetchone():
            # Neue Konfiguration - lösche alte Daten
            cursor.execute('DELETE FROM processed_files')
            cursor.execute('DELETE FROM config')
            
            # Füge neue Konfiguration hinzu
            cursor.execute('''
                INSERT INTO config (config_hash, config_data, created_at)
                VALUES (?, ?, ?)
            ''', (current_hash, json.dumps(CONFIG, indent=2), datetime.now()))
            
            conn.commit()
            print("🗄️ Neue Konfiguration erkannt - Datenbank zurückgesetzt")
        else:
            print("🗄️ Bestehende Konfiguration gefunden - verwende Cache")
        
        conn.close()
        return db_path
        
    except Exception as e:
        print(f"⚠️ Fehler bei Datenbank-Initialisierung: {e}")
        print("🗄️ Erstelle neue Datenbank...")
        
        # Versuche alte Datei zu löschen und neue zu erstellen
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except:
            pass
            
        # Rekursiver Aufruf für neue Datenbank
        return init_database()

def is_file_processed(db_path: str, file_info: FileInfo) -> Optional[Tuple[str, str]]:
    """
    Prüft ob eine Datei bereits mit der aktuellen Konfiguration verarbeitet wurde.
    
    Args:
        db_path: Pfad zur SQLite-Datenbank
        file_info: FileInfo Objekt mit Datei-Metadaten
    
    Returns:
        Optional[Tuple]: (decision, discard_reasons_json) wenn gefunden, sonst None
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        current_hash = get_config_hash()
        
        cursor.execute('''
            SELECT decision, discard_reasons FROM processed_files 
            WHERE file_path = ? AND config_hash = ? AND file_size = ? AND file_mtime = ?
        ''', (file_info.relative_path, current_hash, file_info.size, file_info.mtime))
        
        result = cursor.fetchone()
        conn.close()
        
        return result
    except Exception as e:
        print(f"⚠️ Fehler beim Cache-Zugriff: {e}")
        return None

def save_file_result(db_path: str, file_info: FileInfo, decision: str, discard_reasons: List[str]):
    """
    Speichert das Verarbeitungsergebnis einer Datei in der Datenbank.
    
    Args:
        db_path: Pfad zur SQLite-Datenbank
        file_info: FileInfo Objekt mit Datei-Metadaten
        decision: Entscheidung ('kept' oder 'sorted_out')
        discard_reasons: Liste der Gründe für Aussortierung
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        current_hash = get_config_hash()
        
        # Verwende INSERT OR REPLACE um Duplikate zu vermeiden
        cursor.execute('''
            INSERT OR REPLACE INTO processed_files 
            (file_path, config_hash, file_size, file_mtime, decision, discard_reasons, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (file_info.relative_path, current_hash, file_info.size, file_info.mtime, decision, json.dumps(discard_reasons), datetime.now()))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Fehler beim Speichern in Cache: {e}")

def show_decision_gui(img, detection_result: DetectionResult, filename: str) -> bool:
    """
    Zeigt eine grafische Benutzeroberfläche für manuelle Entscheidungen bei unsicheren Erkennungen.
    
    Args:
        img: OpenCV Bild-Array für die Anzeige
        detection_result: DetectionResult Objekt mit YOLO-Erkennungen
        filename: Name der Datei für Fenstertitel
    
    Returns:
        bool: True wenn Benutzer "BEHALTEN" wählt, False bei "AUSSORTIEREN"
    """
    
    # Filtere nur unsichere Erkennungen für die Anzeige
    uncertain_indices = []
    if detection_result.boxes is not None:
        for i, cls_id in enumerate(detection_result.labels):
            class_name = model.names[int(cls_id)]
            confidence = detection_result.confidences[i]
            x1, y1, x2, y2 = detection_result.boxes[i]
            object_area = (x2 - x1) * (y2 - y1)
            
            # Filtere Objekte, die eine manuelle Entscheidung benötigen
            if (class_name in target_classes and 
                confidence >= min_confidence_threshold and
                ((min_confidence_threshold <= confidence < max_confidence_threshold) or 
                 (small_area_threshold < object_area < large_area_threshold))):
                uncertain_indices.append(i)
    
    # Zeichne nur unsichere Bounding Boxes auf das Bild
    img_display = img.copy()
    if detection_result.boxes is not None:
        for i in uncertain_indices:
            x1, y1, x2, y2 = detection_result.boxes[i].astype(int)
            confidence = detection_result.confidences[i]
            class_name = model.names[int(detection_result.labels[i])]
            area = (x2 - x1) * (y2 - y1)
            
            # Orange für unsichere Objekte
            color = (0, 127, 255)  # Orange
                
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_display, f'{class_name} ({int(area)}) {confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Tkinter GUI erstellen
    root = tk.Tk()
    root.title(f"Entscheidung für: {filename}")
    
    # Maximiere das Fenster oder verwende 90% der Bildschirmgröße
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.9)
    window_height = int(screen_height * 0.9)
    
    root.geometry(f"{window_width}x{window_height}")
    
    # Bild für Tkinter vorbereiten
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    img_pil_original = Image.fromarray(img_rgb)  # Original für Resize speichern
    
    # GUI Elemente
    label = tk.Label(root)
    label.pack(pady=10, expand=True, fill='both')
    
    # Variable für Resize-Timer
    resize_timer = None
    last_size = (0, 0)
    
    def resize_image():
        """Passt das Bild an die aktuelle Fenstergröße an"""
        try:
            # Aktuelle Fenstergröße ermitteln
            current_width = root.winfo_width()
            current_height = root.winfo_height()
            
            # Nur resize wenn sich die Größe signifikant geändert hat
            nonlocal last_size
            if abs(current_width - last_size[0]) < 10 and abs(current_height - last_size[1]) < 10:
                return
            
            last_size = (current_width, current_height)
            
            # Verfügbaren Platz berechnen
            available_width = current_width - 40
            available_height = current_height - 150
            
            if available_width > 50 and available_height > 50:  # Mindestgröße
                # Skalierung berechnen
                img_aspect = img_pil_original.width / img_pil_original.height
                
                if available_width / img_aspect <= available_height:
                    new_width = available_width
                    new_height = int(available_width / img_aspect)
                else:
                    new_height = available_height
                    new_width = int(available_height * img_aspect)
                
                # Bild skalieren und anzeigen
                resized_img = img_pil_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(resized_img)
                label.configure(image=img_tk)
                label.image = img_tk  # Referenz behalten
        except Exception as e:
            print(f"Resize error: {e}")  # Debug-Info
    
    def on_window_resize(event):
        """Handler für Fenstergrößenänderungen mit Timer"""
        nonlocal resize_timer
        if resize_timer:
            root.after_cancel(resize_timer)
        # Warte 100ms nach letzter Änderung bevor resize
        resize_timer = root.after(100, resize_image)
    
    info_text = tk.Label(root, text=f"Datei: {filename}\nOrange=manuelle Entscheidung nötig\nZahlen: (Fläche) Confidence-Score\n\nAngezeigt: mittlere Confidence ({min_confidence_threshold:.2f}-{max_confidence_threshold:.2f}) oder mittlere Objektgröße ({small_area_threshold//1000}k-{large_area_threshold//1000}k)", 
                        font=("Arial", 10))
    info_text.pack(pady=5)
    
    result = {'keep': False}
    
    def on_window_close():
        """Handler für das Schließen des Fensters über X - beendet das gesamte Skript"""
        tqdm.write("\n🛑 Skript durch Benutzer beendet - aktuelle Datei wird nicht verschoben")
        root.destroy()
        exit(0)  # Beendet das gesamte Skript sofort
    
    def keep_file():
        result['keep'] = True
        root.destroy()
    
    def skip_file():
        result['keep'] = False
        root.destroy()
    
    # Event-Handler nur für das Root-Fenster
    root.bind('<Configure>', lambda e: on_window_resize(e) if e.widget == root else None)
    
    # Handler für das Schließen des Fensters über X
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    # Initiales Resize nach kurzer Verzögerung
    root.after(200, resize_image)
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    keep_btn = tk.Button(button_frame, text="BEHALTEN (nicht verschieben)", 
                        command=keep_file, bg="lightgreen", font=("Arial", 12))
    keep_btn.pack(side=tk.LEFT, padx=10)
    
    skip_btn = tk.Button(button_frame, text="AUSSORTIEREN", 
                        command=skip_file, bg="lightcoral", font=("Arial", 12))
    skip_btn.pack(side=tk.LEFT, padx=10)
    
    # Fenster zentrieren
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    return result['keep']

def show_cache_stats(db_path):
    """
    Zeigt detaillierte Statistiken über den Cache-Inhalt an.
    
    Args:
        db_path (str): Pfad zur SQLite-Datenbank
    
    Returns:
        None: Gibt Statistiken auf der Konsole aus
    
    Output:
        - Anzahl gecachter Dateien gesamt
        - Aufschlüsselung nach Entscheidungen (kept/sorted_out)
        - Erstellungsdatum der aktuellen Konfiguration
        - Verkürzter Konfigurations-Hash
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Anzahl verarbeiteter Dateien
        cursor.execute('SELECT COUNT(*) FROM processed_files')
        total_cached = cursor.fetchone()[0]
        
        # Anzahl nach Entscheidung
        cursor.execute('SELECT decision, COUNT(*) FROM processed_files GROUP BY decision')
        decisions = cursor.fetchall()
        
        # Aktueller Konfigurationshash
        current_hash = get_config_hash()
        cursor.execute('SELECT created_at FROM config WHERE config_hash = ?', (current_hash,))
        config_date = cursor.fetchone()
        
        conn.close()
        
        print(f"🗄️ Cache-Statistiken:")
        print(f"   📊 Gesamt gecachte Dateien: {total_cached}")
        for decision, count in decisions:
            emoji = "🗑️" if decision == "sorted_out" else "💾"
            print(f"   {emoji} {decision}: {count}")
        if config_date:
            print(f"   ⚙️ Konfiguration erstellt: {config_date[0]}")
        print(f"   🔑 Konfigurations-Hash: {current_hash[:8]}...")
        
    except Exception as e:
        print(f"⚠️ Fehler beim Lesen der Cache-Statistiken: {e}")

def check_image_basic_criteria(file_info: FileInfo) -> Tuple[List[str], bool]:
    """
    Prüft Grundkriterien wie Auflösung und Seitenverhältnis eines Bildes.
    
    Args:
        file_info: FileInfo Objekt mit Bildabmessungen
    
    Returns:
        Tuple[List[str], bool]: (discard_reasons, should_sort_out)
    """
    img_aspect_ratio = file_info.width / file_info.height
    discard_reasons = []
    
    # Prüfe Mindestauflösung
    if file_info.width < min_resolution[0] or file_info.height < min_resolution[1]:
        discard_reasons.append(f'low_resolution({file_info.width}x{file_info.height})')
        tqdm.write(f"❌ Zu kleine Auflösung aussortiert: {file_info.width}x{file_info.height} < {min_resolution[0]}x{min_resolution[1]}")
        return discard_reasons, True  # should_sort_out = True
    
    # Prüfe Seitenverhältnis
    if abs(img_aspect_ratio - target_aspect_ratio) > aspect_ratio_tolerance:
        discard_reasons.append(f'wrong_aspect_ratio({img_aspect_ratio:.2f})')
        tqdm.write(f"❌ Falsches Seitenverhältnis aussortiert: {img_aspect_ratio:.2f} (erwartet: {target_aspect_ratio:.2f} ±{aspect_ratio_tolerance})")
        return discard_reasons, True  # should_sort_out = True
    
    return discard_reasons, False  # should_sort_out = False

def analyze_yolo_detections(img, file_info: FileInfo) -> DetectionResult:
    """
    Führt YOLO-Analyse durch und bestimmt Aktionen basierend auf Erkennungen.
    
    Args:
        img: OpenCV Bild-Array für YOLO-Analyse
        file_info: FileInfo Objekt mit Datei-Metadaten
    
    Returns:
        DetectionResult: Objekt mit allen Erkennungsdaten und Entscheidungshilfen
    """
    result = DetectionResult()
    
    try:
        yolo_results = model.predict(img, device=device, verbose=False)
    except Exception as e:
        tqdm.write(f"❌ Fehler bei YOLO-Vorhersage für {file_info.relative_path}: {e}")
        result.has_error = True
        return result
    
    if len(yolo_results[0].boxes) > 0:
        result.labels = yolo_results[0].boxes.cls.cpu().numpy()
        result.boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        result.confidences = yolo_results[0].boxes.conf.cpu().numpy()
        
        for i, cls_id in enumerate(result.labels):
            class_name = model.names[int(cls_id)]
            confidence = result.confidences[i]
            
            # Überspringe Erkennungen mit zu niedrigem Confidence-Score
            if confidence < min_confidence_threshold:
                tqdm.write(f"Erkennung übersprungen: {class_name} (confidence: {confidence:.2f} < {min_confidence_threshold})")
                continue
            
            # Nur relevante Klassen berücksichtigen
            if class_name in target_classes:
                x1, y1, x2, y2 = result.boxes[i]
                object_area = (x2 - x1) * (y2 - y1)
                
                if object_area >= large_area_threshold and confidence >= max_confidence_threshold:
                    # Großes Objekt UND hohe Confidence → automatisch aussortieren
                    result.discard_reasons.append(f'{class_name}(area:{int(object_area)},conf:{confidence:.2f})')
                    tqdm.write(f"🔒 Große {class_name} mit hoher Confidence aussortiert: area={int(object_area)}, conf={confidence:.2f}")
                elif object_area <= small_area_threshold:
                    # Kleines Objekt → automatisch behalten
                    tqdm.write(f"✅ Kleine {class_name} behalten: area={int(object_area)}, conf={confidence:.2f}")
                elif confidence <= min_confidence_threshold:
                    # Niedrige Confidence → automatisch behalten
                    tqdm.write(f"✅ {class_name} behalten (niedrige Confidence): area={int(object_area)}, conf={confidence:.2f}")
                else:
                    # Mittlere Confidence ODER mittlere Größe → nachfragen
                    result.needs_manual_decision = True
                    tqdm.write(f"🤔 {class_name} - Entscheidung nötig: area={int(object_area)}, conf={confidence:.2f}")
    
    return result

def make_decision(basic_discard_reasons: List[str], detection_result: DetectionResult, img, file_info: FileInfo, stats: ProcessingStats) -> Tuple[bool, List[str]]:
    """
    Trifft die endgültige Entscheidung basierend auf Analyse-Ergebnissen.
    
    Args:
        basic_discard_reasons: Bereits gefundene Gründe für Aussortierung (aus Grundkriterien)
        detection_result: DetectionResult mit YOLO-Erkennungen
        img: OpenCV Bild-Array für GUI-Anzeige
        file_info: FileInfo Objekt mit Datei-Metadaten
        stats: ProcessingStats Objekt das aktualisiert wird
    
    Returns:
        Tuple[bool, List[str]]: (should_sort_out, all_discard_reasons)
    """
    should_sort_out = False
    all_discard_reasons = basic_discard_reasons + detection_result.discard_reasons
    
    if detection_result.discard_reasons:
        # Sichere Erkennung vorhanden - automatisch aussortieren
        should_sort_out = True
        stats.auto_sorted += 1
        tqdm.write(f"🔒 Automatisches Aussortieren wegen sicherer Erkennung: {detection_result.discard_reasons}")
    elif detection_result.needs_manual_decision:
        # Nur unsichere Erkennungen - GUI für manuelle Entscheidung zeigen
        stats.manual_decisions += 1
        user_wants_to_keep = show_decision_gui(img, detection_result, file_info.relative_path)
        should_sort_out = not user_wants_to_keep
        if should_sort_out:
            stats.manual_sorted += 1
            all_discard_reasons.append('manual_decision_sort_out')
    
    return should_sort_out, all_discard_reasons

def process_file_movement(should_sort_out: bool, file_info: FileInfo, target_folder: str, discard_reasons: List[str], from_cache: bool, needs_manual_decision: bool, stats: ProcessingStats):
    """
    Verarbeitet das Verschieben oder Behalten von Dateien basierend auf der Entscheidung.
    
    Args:
        should_sort_out: True wenn Datei aussortiert werden soll
        file_info: FileInfo Objekt mit Datei-Metadaten
        target_folder: Zielordner für aussortierte Dateien
        discard_reasons: Gründe für die Aussortierung
        from_cache: True wenn Entscheidung aus Cache stammt
        needs_manual_decision: True wenn manuelle Entscheidung getroffen wurde
        stats: ProcessingStats Objekt das aktualisiert wird
    """
    if should_sort_out:
        # Verschiebe unerwünschte Bilder in den Aussortier-Ordner
        target_path = os.path.join(target_folder, file_info.relative_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            shutil.move(file_info.absolute_path, target_path)
            file_uri = f"file://{target_path.replace(' ', '%20')}"
            
            if from_cache:
                tqdm.write(f"🔄 {file_info.relative_path} erneut aussortiert (Cache)")
                stats.auto_sorted += 1
            else:
                tqdm.write(f"🗑️ {file_info.relative_path} aussortiert (gefunden: {discard_reasons})")
                tqdm.write(f"📁 {file_uri}")
                tqdm.write("")
        except Exception as e:
            tqdm.write(f"❌ Fehler beim Verschieben von {file_info.relative_path}: {e}")
            stats.errors += 1
            tqdm.write("")
    else:
        # Datei behalten
        if not from_cache:
            if needs_manual_decision:
                tqdm.write(f"✅ {file_info.relative_path} behalten (manuelle Entscheidung)")
            else:
                tqdm.write(f"✅ {file_info.relative_path} behalten (keine relevanten Objekte erkannt)")
            stats.kept_files += 1

def process_single_image(root: str, filename: str, context: ProcessingContext):
    """
    Verarbeitet eine einzelne Bilddatei komplett von Cache-Prüfung bis zur finalen Entscheidung.
    
    Args:
        root: Wurzelverzeichnis in dem sich die Datei befindet
        filename: Name der Bilddatei
        context: ProcessingContext mit allen notwendigen Verarbeitungsparametern
    """
    # FileInfo Objekt erstellen
    file_info = FileInfo(
        relative_path=os.path.relpath(os.path.join(root, filename), context.source_folder),
        absolute_path=os.path.join(root, filename),
        size=0,
        mtime=0
    )
    
    # Progress Bar Update
    context.pbar.set_postfix({
        'Aktuell': file_info.relative_path[:30] + '...' if len(file_info.relative_path) > 30 else file_info.relative_path,
        'Cache': context.stats.cached_files,
        'Aussortiert': context.stats.auto_sorted,
        'Entscheidungen': context.stats.manual_decisions
    })
    
    # === CACHE-PRÜFUNG ===
    try:
        file_stat = os.stat(file_info.absolute_path)
        file_info.size = file_stat.st_size
        file_info.mtime = file_stat.st_mtime
        
        cached_result = is_file_processed(context.db_path, file_info)
        if cached_result:
            decision, discard_reasons_json = cached_result
            discard_reasons = json.loads(discard_reasons_json) if discard_reasons_json else []
            
            context.stats.cached_files += 1
            should_sort_out = (decision == 'sorted_out')
            
            if should_sort_out:
                tqdm.write(f"🔄 {file_info.relative_path} wird aussortiert (Cache)")
            else:
                tqdm.write(f"💾 {file_info.relative_path} behalten (Cache)")
                context.stats.kept_files += 1
                return  # Frühes Return für gecachte "behalten" Entscheidungen
            
            # Verarbeite gecachte "aussortieren" Entscheidung
            process_file_movement(should_sort_out, file_info, context.target_folder, discard_reasons, True, False, context.stats)
            return
            
    except Exception as e:
        tqdm.write(f"⚠️ Cache-Fehler für {file_info.relative_path}: {e}")
    
    # === NEUE VERARBEITUNG ===
    img = cv2.imread(file_info.absolute_path)
    if img is None:
        tqdm.write(f"⚠️ Kann {file_info.relative_path} nicht laden - überspringe")
        context.stats.errors += 1
        return
    
    context.stats.processed_files += 1
    img_height, img_width = img.shape[:2]
    file_info.width = img_width
    file_info.height = img_height
    
    # Prüfe Grundkriterien (Auflösung, Seitenverhältnis)
    basic_discard_reasons, should_sort_out = check_image_basic_criteria(file_info)
    
    if should_sort_out:
        # Grundkriterien nicht erfüllt - direktes Aussortieren
        context.stats.auto_sorted += 1
        process_file_movement(should_sort_out, file_info, context.target_folder, basic_discard_reasons, False, False, context.stats)
        save_file_result(context.db_path, file_info, 'sorted_out', basic_discard_reasons)
        return
    
    # YOLO-Analyse durchführen
    detection_result = analyze_yolo_detections(img, file_info)
    
    if detection_result.has_error:
        context.stats.errors += 1
        return
    
    # Entscheidung treffen
    should_sort_out, all_discard_reasons = make_decision(basic_discard_reasons, detection_result, img, file_info, context.stats)
    
    # Datei verarbeiten (verschieben oder behalten)
    process_file_movement(should_sort_out, file_info, context.target_folder, all_discard_reasons, False, detection_result.needs_manual_decision, context.stats)
    
    # Ergebnis in Datenbank speichern
    decision = 'sorted_out' if should_sort_out else 'kept'
    save_file_result(context.db_path, file_info, decision, all_discard_reasons)

def main():
    """
    Hauptfunktion für die Bildverarbeitung und -sortierung.
    
    Orchestriert den gesamten Workflow mit vereinfachten Datenstrukturen:
    1. Initialisiert Zielordner und Datenbank
    2. Sammelt alle unterstützten Bilddateien aus dem Quellordner
    3. Verarbeitet jede Datei einzeln mit Cache-Optimierung
    4. Zeigt finale Statistiken an
    """
    # === Aussortier-Ordner anlegen ===
    os.makedirs(target_folder, exist_ok=True)

    # === Datenbank initialisieren ===
    db_path = init_database()
    show_cache_stats(db_path)

    # === Statistiken ===
    stats = ProcessingStats()

    print(f"🔍 Durchsuche Ordner: {source_folder}")
    print(f"📁 Aussortier-Ordner: {target_folder}")
    print(f"🎯 Suche nach: {', '.join(target_classes)}")
    print("=" * 60)

    # === Sammle alle Bilddateien ===
    print("📊 Sammle Bilddateien...")
    all_image_files = []
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename.lower().endswith(CONFIG['supported_extensions']):
                all_image_files.append((root, filename))

    print(f"✅ {len(all_image_files)} Bilddateien gefunden")
    print("=" * 60)

    # === Bilder durchgehen mit Progress Bar ===
    with tqdm(total=len(all_image_files), desc="🖼️ Verarbeite Bilder", unit="Bild") as pbar:
        # ProcessingContext erstellen
        context = ProcessingContext(
            source_folder=source_folder,
            target_folder=target_folder,
            db_path=db_path,
            stats=stats,
            pbar=pbar
        )
        
        for root, filename in all_image_files:
            stats.total_files += 1
            
            # Verarbeite einzelne Datei mit vereinfachter Signatur
            process_single_image(root, filename, context)
            
            # Progress Bar Update
            pbar.update(1)

    # === Abschluss-Statistik ===
    print("\n" + "=" * 60)
    print("📊 VERARBEITUNGSSTATISTIK")
    print("=" * 60)
    print(f"📁 Gefundene Dateien: {stats.total_files}")
    print(f"🗄️ Aus Cache geladen: {stats.cached_files}")
    print(f"✅ Neu verarbeitet: {stats.processed_files}")
    print(f"🔒 Automatisch aussortiert: {stats.auto_sorted}")
    print(f"🤔 Manuelle Entscheidungen: {stats.manual_decisions}")
    print(f"   └─ 🗑️ Manuell aussortiert: {stats.manual_sorted}")
    print(f"   └─ 💾 Manuell behalten: {stats.manual_decisions - stats.manual_sorted}")
    print(f"💾 Behalten gesamt: {stats.kept_files}")
    print(f"❌ Fehler: {stats.errors}")
    print(f"🗑️ Aussortiert gesamt: {stats.auto_sorted + stats.manual_sorted}")
    print("=" * 60)
    print(f"💾 Cache-Datei: {os.path.join(source_folder, '.image_filter_cache.db')}")
    print("✨ Verarbeitung abgeschlossen!")

if __name__ == "__main__":
    main()
