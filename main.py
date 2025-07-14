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
    """Erstellt einen Hash der aktuellen Konfiguration"""
    config_str = json.dumps(CONFIG, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def init_database():
    """Initialisiert die SQLite-Datenbank im Source-Ordner"""
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

def is_file_processed(db_path, file_path, file_size, file_mtime):
    """Prüft ob eine Datei bereits mit der aktuellen Konfiguration verarbeitet wurde"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        current_hash = get_config_hash()
        
        cursor.execute('''
            SELECT decision, discard_reasons FROM processed_files 
            WHERE file_path = ? AND config_hash = ? AND file_size = ? AND file_mtime = ?
        ''', (file_path, current_hash, file_size, file_mtime))
        
        result = cursor.fetchone()
        conn.close()
        
        return result
    except Exception as e:
        print(f"⚠️ Fehler beim Cache-Zugriff: {e}")
        return None

def save_file_result(db_path, file_path, file_size, file_mtime, decision, discard_reasons):
    """Speichert das Verarbeitungsergebnis einer Datei"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        current_hash = get_config_hash()
        
        # Verwende INSERT OR REPLACE um Duplikate zu vermeiden
        cursor.execute('''
            INSERT OR REPLACE INTO processed_files 
            (file_path, config_hash, file_size, file_mtime, decision, discard_reasons, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (file_path, current_hash, file_size, file_mtime, decision, json.dumps(discard_reasons), datetime.now()))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Fehler beim Speichern in Cache: {e}")

def show_decision_gui(img, boxes, labels, confidences, filename):
    """Zeigt GUI für manuelle Entscheidung bei unsicheren Erkennungen"""
    
    # Filtere nur unsichere Erkennungen für die Anzeige
    uncertain_indices = []
    for i, cls_id in enumerate(labels):
        class_name = model.names[int(cls_id)]
        confidence = confidences[i]
        x1, y1, x2, y2 = boxes[i]
        object_area = (x2 - x1) * (y2 - y1)
        
        # Filtere Objekte, die eine manuelle Entscheidung benötigen
        if (class_name in target_classes and 
            confidence >= min_confidence_threshold and
            ((min_confidence_threshold <= confidence < max_confidence_threshold) or 
             (small_area_threshold < object_area < large_area_threshold))):
            uncertain_indices.append(i)
    
    # Zeichne nur unsichere Bounding Boxes auf das Bild
    img_display = img.copy()
    for i in uncertain_indices:
        x1, y1, x2, y2 = boxes[i].astype(int)
        confidence = confidences[i]
        class_name = model.names[int(labels[i])]
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
    """Zeigt Statistiken über den Cache"""
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

def main():
    """Hauptfunktion für die Bildverarbeitung"""
    # === Aussortier-Ordner anlegen ===
    os.makedirs(target_folder, exist_ok=True)

    # === Datenbank initialisieren ===
    db_path = init_database()
    show_cache_stats(db_path)

    # === Statistiken ===
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'cached_files': 0,
        'auto_sorted': 0,
        'manual_decisions': 0,
        'manual_sorted': 0,  # Aus manuellen Entscheidungen aussortiert
        'kept_files': 0,
        'errors': 0
    }

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
        for root, filename in all_image_files:
            stats['total_files'] += 1
            relative_path = os.path.relpath(os.path.join(root, filename), source_folder)
            
            # Progress Bar Update
            pbar.set_postfix({
                'Aktuell': relative_path[:30] + '...' if len(relative_path) > 30 else relative_path,
                'Cache': stats['cached_files'],
                'Aussortiert': stats['auto_sorted'],
                'Entscheidungen': stats['manual_decisions']
            })

            image_path = os.path.join(root, filename)
            
            # Prüfe Cache/Datenbank
            try:
                file_stat = os.stat(image_path)
                file_size = file_stat.st_size
                file_mtime = file_stat.st_mtime
                
                cached_result = is_file_processed(db_path, relative_path, file_size, file_mtime)
                if cached_result:
                    decision, discard_reasons_json = cached_result
                    discard_reasons = json.loads(discard_reasons_json) if discard_reasons_json else []
                    
                    stats['cached_files'] += 1
                    
                    # Progress Bar Update
                    pbar.set_postfix({
                        'Aktuell': relative_path[:30] + '...' if len(relative_path) > 30 else relative_path,
                        'Cache': stats['cached_files'],
                        'Aussortiert': stats['auto_sorted'],
                        'Entscheidungen': stats['manual_decisions']
                    })
                    
                    # Setze Variablen für einheitliche Verarbeitung am Ende
                    should_sort_out = (decision == 'sorted_out')
                    needs_manual_decision = False
                    from_cache = True
                    
                    if should_sort_out:
                        tqdm.write(f"🔄 {relative_path} wird aussortiert (Cache)")
                    else:
                        tqdm.write(f"💾 {relative_path} behalten (Cache)")
                        stats['kept_files'] += 1
                        pbar.update(1)
                        continue
                else:
                    # Datei nicht im Cache - normale Verarbeitung
                    from_cache = False
                    
            except Exception as e:
                tqdm.write(f"⚠️ Cache-Fehler für {relative_path}: {e}")
                from_cache = False
            
            # Nur verarbeiten wenn nicht aus Cache
            if not from_cache:
                img = cv2.imread(image_path)
                if img is None:
                    tqdm.write(f"⚠️ Kann {relative_path} nicht laden - überspringe")
                    stats['errors'] += 1
                    pbar.update(1)
                    continue

                # Prüfe Bildauflösung und Seitenverhältnis
                img_height, img_width = img.shape[:2]
                img_aspect_ratio = img_width / img_height
                
                discard_reasons = []
                needs_manual_decision = False
                
                # Prüfe Mindestauflösung
                if img_width < min_resolution[0] or img_height < min_resolution[1]:
                    discard_reasons.append(f'low_resolution({img_width}x{img_height})')
                    tqdm.write(f"❌ Zu kleine Auflösung aussortiert: {img_width}x{img_height} < {min_resolution[0]}x{min_resolution[1]}")
                # Prüfe Seitenverhältnis
                elif abs(img_aspect_ratio - target_aspect_ratio) > aspect_ratio_tolerance:
                    discard_reasons.append(f'wrong_aspect_ratio({img_aspect_ratio:.2f})')
                    tqdm.write(f"❌ Falsches Seitenverhältnis aussortiert: {img_aspect_ratio:.2f} (erwartet: {target_aspect_ratio:.2f} ±{aspect_ratio_tolerance})")
                else:
                    # Nur bei qualifizierten Bildern YOLO-Analyse durchführen
                    try:
                        results = model.predict(img, device=device, verbose=False)
                    except Exception as e:
                        tqdm.write(f"❌ Fehler bei YOLO-Vorhersage für {relative_path}: {e}")
                        stats['errors'] += 1
                        pbar.update(1)
                        continue
                        
                    stats['processed_files'] += 1
                    
                    # Prüfe auf relevante Objekte
                    if len(results[0].boxes) > 0:
                        labels = results[0].boxes.cls.cpu().numpy()
                        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
                        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
                        
                        for i, cls_id in enumerate(labels):
                            class_name = model.names[int(cls_id)]
                            confidence = confidences[i]
                            
                            # Überspringe Erkennungen mit zu niedrigem Confidence-Score
                            if confidence < min_confidence_threshold:
                                tqdm.write(f"Erkennung übersprungen: {class_name} (confidence: {confidence:.2f} < {min_confidence_threshold})")
                                continue
                            
                            # Nur relevante Klassen berücksichtigen
                            if class_name in target_classes:
                                # Prüfe Größe des Objekts (Bounding Box)
                                x1, y1, x2, y2 = boxes[i]
                                object_area = (x2 - x1) * (y2 - y1)
                                
                                if object_area >= large_area_threshold and confidence >= max_confidence_threshold:
                                    # Großes Objekt UND hohe Confidence → automatisch aussortieren
                                    discard_reasons.append(f'{class_name}(area:{int(object_area)},conf:{confidence:.2f})')
                                    tqdm.write(f"🔒 Große {class_name} mit hoher Confidence aussortiert: area={int(object_area)}, conf={confidence:.2f}")
                                elif object_area <= small_area_threshold:
                                    # Kleines Objekt → automatisch behalten
                                    tqdm.write(f"✅ Kleine {class_name} behalten: area={int(object_area)}, conf={confidence:.2f}")
                                elif confidence <= min_confidence_threshold:
                                    # Niedrige Confidence → automatisch behalten
                                    tqdm.write(f"✅ {class_name} behalten (niedrige Confidence): area={int(object_area)}, conf={confidence:.2f}")
                                else:
                                    # Mittlere Confidence ODER mittlere Größe → nachfragen
                                    needs_manual_decision = True
                                    tqdm.write(f"🤔 {class_name} - Entscheidung nötig: area={int(object_area)}, conf={confidence:.2f}")

                # Entscheidung treffen
                should_sort_out = False
                
                if discard_reasons:
                    # Sichere Erkennung vorhanden - automatisch aussortieren (auch wenn zusätzlich unsichere Erkennungen da sind)
                    should_sort_out = True
                    stats['auto_sorted'] += 1
                    tqdm.write(f"🔒 Automatisches Aussortieren wegen sicherer Erkennung: {discard_reasons}")
                elif needs_manual_decision:
                    # Nur unsichere Erkennungen - GUI für manuelle Entscheidung zeigen
                    # Wenn keep_file() aufgerufen wird, wird result['keep'] = True gesetzt
                    # Das bedeutet: NICHT aussortieren (im Quellordner lassen)
                    stats['manual_decisions'] += 1
                    user_wants_to_keep = show_decision_gui(img, boxes, labels, confidences, relative_path)
                    should_sort_out = not user_wants_to_keep  # Umkehrung!
                    if should_sort_out:
                        stats['manual_sorted'] += 1
                        discard_reasons.append('manual_decision_sort_out')

            # === EINHEITLICHE DATEI-VERARBEITUNG ===
            # Hier ist der einzige Ort wo Dateien verschoben werden
            if should_sort_out:
                # Verschiebe unerwünschte Bilder in den Aussortier-Ordner
                # Behalte die relative Pfadstruktur bei
                target_path = os.path.join(target_folder, relative_path)
                
                # Erstelle Zielverzeichnis falls nötig
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                try:
                    shutil.move(image_path, target_path)
                    # Erstelle clickbaren Link mit file:// URI
                    file_uri = f"file://{target_path.replace(' ', '%20')}"
                    
                    if from_cache:
                        tqdm.write(f"🔄 {relative_path} erneut aussortiert (Cache)")
                        stats['auto_sorted'] += 1
                    else:
                        tqdm.write(f"🗑️ {relative_path} aussortiert (gefunden: {discard_reasons})")
                        tqdm.write(f"📁 {file_uri}")
                        tqdm.write("")
                    
                    # Speichere Ergebnis in Datenbank (nur für neue Verarbeitungen)
                    if not from_cache:
                        save_file_result(db_path, relative_path, file_size, file_mtime, 'sorted_out', discard_reasons)
                        
                except Exception as e:
                    tqdm.write(f"❌ Fehler beim Verschieben von {relative_path}: {e}")
                    stats['errors'] += 1
                    tqdm.write("")
            else:
                # Datei behalten
                if from_cache:
                    # Bereits aus Cache verarbeitet - nichts weiter zu tun
                    pass
                elif needs_manual_decision:
                    tqdm.write(f"✅ {relative_path} behalten (manuelle Entscheidung)")
                    stats['kept_files'] += 1
                    tqdm.write("")
                    # Speichere Ergebnis in Datenbank
                    save_file_result(db_path, relative_path, file_size, file_mtime, 'kept', [])
                else:
                    # Keine Erkennungen - Bild bleibt im Quellordner
                    tqdm.write(f"✅ {relative_path} behalten (keine relevanten Objekte erkannt)")
                    stats['kept_files'] += 1
                    # Speichere Ergebnis in Datenbank
                    save_file_result(db_path, relative_path, file_size, file_mtime, 'kept', [])
            
            # Progress Bar Update
            pbar.update(1)

    # === Abschluss-Statistik ===
    print("\n" + "=" * 60)
    print("📊 VERARBEITUNGSSTATISTIK")
    print("=" * 60)
    print(f"📁 Gefundene Dateien: {stats['total_files']}")
    print(f"🗄️ Aus Cache geladen: {stats['cached_files']}")
    print(f"✅ Neu verarbeitet: {stats['processed_files']}")
    print(f"🔒 Automatisch aussortiert: {stats['auto_sorted']}")
    print(f"🤔 Manuelle Entscheidungen: {stats['manual_decisions']}")
    print(f"   └─ 🗑️ Manuell aussortiert: {stats['manual_sorted']}")
    print(f"   └─ 💾 Manuell behalten: {stats['manual_decisions'] - stats['manual_sorted']}")
    print(f"💾 Behalten gesamt: {stats['kept_files']}")
    print(f"❌ Fehler: {stats['errors']}")
    print(f"🗑️ Aussortiert gesamt: {stats['auto_sorted'] + stats['manual_sorted']}")
    print("=" * 60)
    print(f"💾 Cache-Datei: {os.path.join(source_folder, '.image_filter_cache.db')}")
    print("✨ Verarbeitung abgeschlossen!")

if __name__ == "__main__":
    main()
