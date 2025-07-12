#!/usr/bin/env python3
"""
YOLO Klassen-Anzeige Tool
Zeigt alle verfügbaren Objektklassen des YOLOv8 Modells an.
"""

import os
from ultralytics import YOLO

def show_yolo_classes(model_path="yolov8s.pt"):
    """Zeigt alle verfügbaren YOLO-Klassen an"""
    
    print("🔍 Lade YOLO-Modell...")
    
    try:
        # Modell laden
        model = YOLO(model_path)
        
        print(f"✅ Modell geladen: {model_path}")
        print(f"📊 Anzahl Klassen: {len(model.names)}")
        print("\n" + "=" * 60)
        print("📋 VERFÜGBARE OBJEKTKLASSEN")
        print("=" * 60)
        
        # Alle Klassen mit Index anzeigen
        for class_id, class_name in model.names.items():
            print(f"{class_id:3d}: {class_name}")
        
        print("=" * 60)
        
        # Zusätzliche Informationen
        print(f"\n📝 Modell-Info:")
        print(f"   - Typ: {type(model.model).__name__}")
        print(f"   - Eingabegröße: {getattr(model.model, 'imgsz', 'Unbekannt')}")
        
        # Häufig verwendete Klassen hervorheben
        common_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 
                         'dog', 'cat', 'bird', 'horse', 'cow', 'sheep']
        
        available_common = [name for name in common_classes if name in model.names.values()]
        
        if available_common:
            print(f"\n🎯 Häufig verwendete verfügbare Klassen:")
            for class_name in available_common:
                # Finde die ID der Klasse
                class_id = next(id for id, name in model.names.items() if name == class_name)
                print(f"   - {class_name} (ID: {class_id})")
        
        # Beispiel für target_classes
        print(f"\n💡 Beispiel für CONFIG['target_classes']:")
        example_classes = available_common[:5] if available_common else list(model.names.values())[:5]
        print(f"   'target_classes': {example_classes}")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        print(f"   Stelle sicher, dass die Datei '{model_path}' existiert.")
        return False
    
    return True

def main():
    """Hauptfunktion"""
    print("🤖 YOLO Klassen-Anzeige Tool")
    print("=" * 60)
    
    # Standardmodellpfad aus der Hauptkonfiguration
    model_path = "yolov8s.pt"
    
    # Prüfe ob Modell existiert
    if not os.path.exists(model_path):
        print(f"⚠️ Modell '{model_path}' nicht gefunden.")
        
        # Versuche andere Modellvarianten
        alternative_models = ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        found_model = None
        
        for alt_model in alternative_models:
            if os.path.exists(alt_model):
                found_model = alt_model
                break
        
        if found_model:
            print(f"✅ Verwende stattdessen: {found_model}")
            model_path = found_model
        else:
            print("❌ Kein YOLO-Modell gefunden.")
            print("💡 Lade ein Modell automatisch herunter...")
            model_path = "yolov8s.pt"  # YOLO lädt automatisch herunter
    
    # Klassen anzeigen
    success = show_yolo_classes(model_path)
    
    if success:
        print(f"\n✨ Fertig! Du kannst diese Klassen in der CONFIG['target_classes'] verwenden.")
    else:
        print(f"\n❌ Konnte Klassen nicht anzeigen.")

if __name__ == "__main__":
    main()
