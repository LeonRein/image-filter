import os
import shutil
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from tqdm import tqdm

# === Konfiguration ===
CONFIG = {
    'source_folder': "/home/hans/wallpaper/gallery-dl/",
    'target_folder': "/home/hans/wallpaper/discarded/",  # Ordner für unerwünschte Bilder
    'model_path': "yolov8s.pt",  # vortrainiertes YOLOv8-Modell
    'device': 'cpu',
    'target_classes': ['car', 'person', 'motorcycle'],  # Zielklassen: Autos und erkennbare Personen
    'large_area_threshold': 300000,  # Mindestfläche für große Objekte (≥200k + hohe Confidence = aussortieren)
    'small_area_threshold': 50000,  # Maximale Fläche für kleine Objekte (≤75k = automatisch behalten)
    'min_confidence_threshold': 0.25,  # Mindest-Confidence-Score (darunter wird ignoriert)
    'max_confidence_threshold': 0.7,  # Maximaler Confidence-Schwelle (≥0.7 bei großen Objekten = aussortieren)
    'supported_extensions': (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
}

# Extrahiere Konfigurationswerte für Kompatibilität
source_folder = CONFIG['source_folder']
target_folder = CONFIG['target_folder']
model_path = CONFIG['model_path']
device = CONFIG['device']
target_classes = CONFIG['target_classes']
large_area_threshold = CONFIG['large_area_threshold']
small_area_threshold = CONFIG['small_area_threshold']
min_confidence_threshold = CONFIG['min_confidence_threshold']
max_confidence_threshold = CONFIG['max_confidence_threshold']

# === Modell laden ===
model = YOLO(model_path)

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

# === Aussortier-Ordner anlegen ===
os.makedirs(target_folder, exist_ok=True)

# === Statistiken ===
stats = {
    'total_files': 0,
    'processed_files': 0,
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
            'Aussortiert': stats['auto_sorted'],
            'Entscheidungen': stats['manual_decisions']
        })

        image_path = os.path.join(root, filename)
        img = cv2.imread(image_path)
        if img is None:
            tqdm.write(f"⚠️ Kann {relative_path} nicht laden - überspringe")
            stats['errors'] += 1
            pbar.update(1)
            continue

        try:
            results = model.predict(img, device=device, verbose=False)
        except Exception as e:
            tqdm.write(f"❌ Fehler bei YOLO-Vorhersage für {relative_path}: {e}")
            stats['errors'] += 1
            pbar.update(1)
            continue
            
        stats['processed_files'] += 1
        
        detected_objects = []
        needs_manual_decision = False
        
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
                        detected_objects.append(f'{class_name}(area:{int(object_area)},conf:{confidence:.2f})')
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
                        detected_objects.append(f'{class_name}(area:{int(object_area)},conf:{confidence:.2f})')
                        tqdm.write(f"🤔 {class_name} - Entscheidung nötig: area={int(object_area)}, conf={confidence:.2f}")

        # Entscheidung treffen
        should_sort_out = False
        
        if detected_objects:
            # Sichere Erkennung vorhanden - automatisch aussortieren (auch wenn zusätzlich unsichere Erkennungen da sind)
            should_sort_out = True
            stats['auto_sorted'] += 1
            tqdm.write(f"🔒 Automatisches Aussortieren wegen sicherer Erkennung: {detected_objects}")
        elif needs_manual_decision:
            # Nur unsichere Erkennungen - GUI für manuelle Entscheidung zeigen
            # Wenn keep_file() aufgerufen wird, wird result['keep'] = True gesetzt
            # Das bedeutet: NICHT aussortieren (im Quellordner lassen)
            stats['manual_decisions'] += 1
            user_wants_to_keep = show_decision_gui(img, boxes, labels, confidences, relative_path)
            should_sort_out = not user_wants_to_keep  # Umkehrung!
            if should_sort_out:
                stats['manual_sorted'] += 1
                detected_objects.append('manual_decision_sort_out')

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
                tqdm.write(f"🗑️ {relative_path} aussortiert (gefunden: {detected_objects})")
                tqdm.write(f"📁 {file_uri}")
                tqdm.write("")
            except Exception as e:
                tqdm.write(f"❌ Fehler beim Verschieben von {relative_path}: {e}")
                stats['errors'] += 1
                tqdm.write("")
        elif needs_manual_decision:
            tqdm.write(f"✅ {relative_path} behalten (manuelle Entscheidung)")
            stats['kept_files'] += 1
            tqdm.write("")
        else:
            # Keine Erkennungen - Bild bleibt im Quellordner
            tqdm.write(f"✅ {relative_path} behalten (keine relevanten Objekte erkannt)")
            stats['kept_files'] += 1
        
        # Progress Bar Update
        pbar.update(1)

# === Abschluss-Statistik ===
print("\n" + "=" * 60)
print("📊 VERARBEITUNGSSTATISTIK")
print("=" * 60)
print(f"📁 Gefundene Dateien: {stats['total_files']}")
print(f"✅ Erfolgreich verarbeitet: {stats['processed_files']}")
print(f"🔒 Automatisch aussortiert: {stats['auto_sorted']}")
print(f"🤔 Manuelle Entscheidungen: {stats['manual_decisions']}")
print(f"   └─ �️ Manuell aussortiert: {stats['manual_sorted']}")
print(f"   └─ 💾 Manuell behalten: {stats['manual_decisions'] - stats['manual_sorted']}")
print(f"💾 Behalten gesamt: {stats['kept_files']}")
print(f"❌ Fehler: {stats['errors']}")
print(f"🗑️ Aussortiert gesamt: {stats['auto_sorted'] + stats['manual_sorted']}")
print("=" * 60)
print("✨ Verarbeitung abgeschlossen!")
