#!/usr/bin/env python3
"""
Test der FastFaceRecognition Klasse mit dem hochgeladenen Gesicht
"""
import sys
import os
sys.path.append('/Users/jonasgartner/Documents/GitHub/HomeShield_AI/Webinterface')

from app import FastFaceRecognition
import cv2
import numpy as np
from PIL import Image

def test_fast_face_recognition():
    print("🔍 Testing FastFaceRecognition Class")
    print("=" * 50)
    
    # Initialisiere FastFaceRecognition
    face_recognizer = FastFaceRecognition()
    
    print(f"Geladene Gesichter: {len(face_recognizer.known_faces)}")
    
    for i, face in enumerate(face_recognizer.known_faces):
        print(f"  {i+1}. {face['name']}")
    
    if len(face_recognizer.known_faces) == 0:
        print("❌ Keine Gesichter geladen!")
        return False
    
    # Test mit dem hochgeladenen Bild selbst
    test_image_path = "/Users/jonasgartner/Documents/GitHub/HomeShield_AI/Webinterface/static/faces/7c15827c-5dc9-40d1-908f-0729930d0116.jpg"
    
    print(f"\n🖼️  Teste mit eigenem Bild: {os.path.basename(test_image_path)}")
    
    try:
        # Lade Testbild
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ Bild konnte nicht geladen werden")
            return False
        
        print(f"✅ Bild geladen: {image.shape}")
        
        # Teste Erkennung
        result = face_recognizer.detect_faces_in_image(image)
        
        print(f"\n📊 Erkennungsergebnis:")
        print(f"   Gefundene Gesichter: {result['total_faces']}")
        
        for i, face in enumerate(result['faces']):
            print(f"   Gesicht {i+1}:")
            print(f"     Name: {face['name']}")
            print(f"     Bekannt: {'✅' if face['is_known'] else '❌'}")
            print(f"     Confidence: {face['confidence']:.4f}")
            print(f"     Position: {face['location']}")
        
        # Erfolg wenn Jonas erkannt wurde
        jonas_detected = any(face['name'] == 'Jonas' and face['is_known'] for face in result['faces'])
        
        if jonas_detected:
            print("\n✅ Jonas wurde erfolgreich erkannt!")
            return True
        else:
            print("\n❌ Jonas wurde NICHT erkannt!")
            return False
            
    except Exception as e:
        print(f"❌ Fehler beim Testen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_modified_image():
    """Teste mit leicht veränderten Bildern"""
    print("\n🔄 Test mit veränderten Bildern")
    print("=" * 40)
    
    face_recognizer = FastFaceRecognition()
    original_path = "/Users/jonasgartner/Documents/GitHub/HomeShield_AI/Webinterface/static/faces/7c15827c-5dc9-40d1-908f-0729930d0116.jpg"
    
    # Original Bild laden
    original = cv2.imread(original_path)
    if original is None:
        print("❌ Original nicht gefunden")
        return
    
    # Verschiedene Transformationen testen
    tests = [
        ("Original", original),
        ("90% Größe", cv2.resize(original, None, fx=0.9, fy=0.9)),
        ("110% Größe", cv2.resize(original, None, fx=1.1, fy=1.1)),
        ("Heller (+30)", cv2.add(original, np.ones(original.shape, dtype=np.uint8) * 30)),
        ("Dunkler (-30)", cv2.subtract(original, np.ones(original.shape, dtype=np.uint8) * 30))
    ]
    
    for test_name, test_image in tests:
        print(f"\n  🔍 {test_name}:")
        try:
            result = face_recognizer.detect_faces_in_image(test_image)
            jonas_found = any(f['name'] == 'Jonas' for f in result['faces'])
            
            if result['total_faces'] > 0:
                best_confidence = max(f['confidence'] for f in result['faces'])
                print(f"    Gesichter: {result['total_faces']}, Jonas: {'✅' if jonas_found else '❌'}, Confidence: {best_confidence:.3f}")
            else:
                print(f"    Keine Gesichter erkannt")
                
        except Exception as e:
            print(f"    ❌ Fehler: {e}")

if __name__ == "__main__":
    print("🚀 FastFaceRecognition Test")
    print("=" * 60)
    
    success = test_fast_face_recognition()
    
    if success:
        test_with_modified_image()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test erfolgreich - Gesichtserkennung funktioniert!")
    else:
        print("❌ Test fehlgeschlagen - Problem bei der Gesichtserkennung")