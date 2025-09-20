#!/usr/bin/env python3
"""
Debug-Script für Gesichtserkennung
Testet die Funktionalität der FastFaceRecognition Klasse
"""
import os
import sys
import json
import face_recognition as fr
import numpy as np
from PIL import Image

# Pfade
faces_json_path = "/Users/jonasgartner/Documents/GitHub/HomeShield_AI/Gesichtserkennung/bekannte_gesichter.json"
faces_dir = "/Users/jonasgartner/Documents/GitHub/HomeShield_AI/Webinterface/static/faces"

def test_face_recognition():
    print("🔍 Face Recognition Debug Test")
    print("=" * 50)
    
    # 1. Prüfe ob JSON existiert
    print(f"1. JSON Datei prüfen: {faces_json_path}")
    if not os.path.exists(faces_json_path):
        print("❌ bekannte_gesichter.json nicht gefunden!")
        return False
    
    # 2. Lade JSON
    try:
        with open(faces_json_path, 'r', encoding='utf-8') as f:
            faces_data = json.load(f)
        print(f"✅ JSON geladen: {len(faces_data)} Einträge")
        print(f"   Daten: {faces_data}")
    except Exception as e:
        print(f"❌ JSON Fehler: {e}")
        return False
    
    # 3. Teste jedes Gesicht
    known_faces = []
    
    for i, face_data in enumerate(faces_data):
        name = face_data['Name']
        image_file = face_data['Image']
        image_path = os.path.join(faces_dir, image_file)
        
        print(f"\n2.{i+1} Teste Gesicht: {name}")
        print(f"   Bild: {image_file}")
        print(f"   Pfad: {image_path}")
        
        # Prüfe ob Datei existiert
        if not os.path.exists(image_path):
            print(f"❌ Bilddatei nicht gefunden!")
            continue
        
        file_size = os.path.getsize(image_path)
        print(f"   Dateigröße: {file_size} bytes")
        
        try:
            # Lade mit face_recognition
            image = fr.load_image_file(image_path)
            print(f"   ✅ Bild geladen: {image.shape}")
            
            # Finde Gesichter
            face_locations = fr.face_locations(image)
            print(f"   Gefundene Gesichter: {len(face_locations)}")
            
            if len(face_locations) == 0:
                print(f"   ❌ Kein Gesicht im Bild erkannt!")
                
                # Versuche andere Parameter
                print("   🔄 Teste mit CNN-Modell...")
                face_locations_cnn = fr.face_locations(image, model="cnn")
                print(f"   CNN Gesichter: {len(face_locations_cnn)}")
                
                if len(face_locations_cnn) > 0:
                    face_locations = face_locations_cnn
                    print("   ✅ CNN Modell erfolgreich!")
            
            if len(face_locations) > 0:
                # Erstelle Encodings
                encodings = fr.face_encodings(image, face_locations)
                print(f"   Encodings erstellt: {len(encodings)}")
                
                if encodings:
                    encoding = encodings[0]
                    print(f"   ✅ Encoding erfolgreich: {encoding.shape}")
                    
                    known_faces.append({
                        'name': name,
                        'encoding': encoding,
                        'location': face_locations[0]
                    })
                else:
                    print(f"   ❌ Encoding fehlgeschlagen!")
            
        except Exception as e:
            print(f"   ❌ Fehler beim Verarbeiten: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n3. Zusammenfassung:")
    print(f"   Erfolgreich geladene Gesichter: {len(known_faces)}")
    
    if known_faces:
        print("\n4. Test der Selbsterkennung:")
        for face in known_faces:
            name = face['name']
            encoding = face['encoding']
            
            # Teste gegen sich selbst
            distance = fr.face_distance([encoding], encoding)[0]
            print(f"   {name}: Selbst-Distanz = {distance:.4f}")
            
            # Teste Toleranzen
            tolerances = [0.3, 0.4, 0.5, 0.6, 0.7]
            for tolerance in tolerances:
                match = fr.compare_faces([encoding], encoding, tolerance=tolerance)[0]
                print(f"   {name}: Toleranz {tolerance} = {'✅' if match else '❌'}")
    
    return len(known_faces) > 0

def test_image_formats():
    """Teste verschiedene Bildformate"""
    print("\n🖼️  Bildformat Tests")
    print("=" * 30)
    
    test_image_path = os.path.join(faces_dir, "7c15827c-5dc9-40d1-908f-0729930d0116.jpg")
    
    if not os.path.exists(test_image_path):
        print("❌ Test-Bild nicht gefunden")
        return
    
    try:
        # Mit PIL laden
        pil_image = Image.open(test_image_path)
        print(f"PIL: {pil_image.format}, {pil_image.mode}, {pil_image.size}")
        
        # Zu RGB konvertieren falls nötig
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            print("Zu RGB konvertiert")
        
        # Zu numpy array
        np_image = np.array(pil_image)
        print(f"NumPy: {np_image.shape}, dtype={np_image.dtype}")
        
        # Mit face_recognition testen
        faces = fr.face_locations(np_image)
        print(f"Erkannte Gesichter: {len(faces)}")
        
        if len(faces) > 0:
            encodings = fr.face_encodings(np_image, faces)
            print(f"Encodings: {len(encodings)}")
            
        return len(faces) > 0
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 HomeShield AI - Face Recognition Debug")
    print("=" * 60)
    
    # Prüfe face_recognition Installation
    try:
        print(f"face_recognition Version: {fr.__version__ if hasattr(fr, '__version__') else 'unbekannt'}")
    except:
        print("face_recognition nicht korrekt installiert!")
    
    # Haupttest
    success = test_face_recognition()
    
    # Bildformat-Test
    format_success = test_image_formats()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Face Recognition funktioniert grundsätzlich!")
    else:
        print("❌ Face Recognition hat Probleme!")
    
    if format_success:
        print("✅ Bildformat-Handling funktioniert!")
    else:
        print("❌ Bildformat-Probleme erkannt!")