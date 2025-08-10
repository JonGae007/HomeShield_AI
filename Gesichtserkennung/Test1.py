from deepface import DeepFace
import cv2
import os
import json # Importiere das json-Modul
from playsound import playsound
import time # Importiere das time-Modul
import requests # Für Pushover-Benachrichtigungen

camera = cv2.VideoCapture(0)

# Hauptschleife für kontinuierliche Gesichtserkennung
while True:
    return_value, image = camera.read()
    # Speichere das Bild temporär oder verarbeite es direkt
    live_view_path = os.path.join(r'C:\Users\jonas\Documents\GitHub\HomeShield_AI\Gesichtserkennung\Tests' , 'live_view.jpg')
    cv2.imwrite(live_view_path, image)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = live_view_path # Verwende den Pfad zum gerade aufgenommenen Bild

    # Lade die bekannten Gesichter aus der JSON-Datei
    json_file_path = os.path.join(script_dir, "bekannte_gesichter.json")
    known_face_names = []
    known_face_imgs_relative = []

    try:
        with open(json_file_path, 'r') as f:
            gesichter_daten = json.load(f)
            for eintrag in gesichter_daten:
                known_face_names.append(eintrag["Name"])
                known_face_imgs_relative.append(eintrag["Image"])
        if not known_face_names:
            print(f"Warnung: Keine Gesichter in {json_file_path} gefunden. Bitte zuerst Kalibration.py ausführen.")
            # Optional: Beenden oder Standardwerte verwenden
            # exit() 
    except FileNotFoundError:
        print(f"Fehler: Kalibrationsdatei {json_file_path} nicht gefunden. Bitte zuerst Kalibration.py ausführen.")
        exit()
    except json.JSONDecodeError:
        print(f"Fehler: Kalibrationsdatei {json_file_path} ist nicht korrekt formatiert.")
        exit()


    known_face_imgs = [os.path.join(script_dir, name) for name in known_face_imgs_relative]

    best_match_name = None
    min_distance = float('inf')
    detector_to_use = 'mtcnn'
    model_to_use = 'Facenet512'

    print(f"Versuche Gesichter mit Detektor '{detector_to_use}' und Modell '{model_to_use}' zu vergleichen.")

    for i, known_img_path in enumerate(known_face_imgs):
        try:
            result = DeepFace.verify(
                img1_path=img_path, 
                img2_path=known_img_path, 
                detector_backend=detector_to_use,
                model_name=model_to_use,
                enforce_detection=True
            ) 
            
            print(f"Vergleich von '{os.path.basename(img_path)}' mit '{os.path.basename(known_img_path)}' ({known_face_names[i]}):")
            print(f"  Ergebnis: {result}")

            if result["verified"] and result["distance"] < min_distance:
                min_distance = result["distance"]
                best_match_name = known_face_names[i]
                
        except Exception as e:
            print(f"Fehler beim Vergleichen mit {known_face_names[i]} ({os.path.basename(known_img_path)}): {e}")

    if best_match_name:
        print(f"Das Gesicht in {os.path.basename(img_path)} gehört am ehesten zu {best_match_name} (Distanz: {min_distance:.4f})")
    else:
        print(f"Keine Übereinstimmung für {os.path.basename(img_path)} gefunden mit den aktuellen Einstellungen.")

    # Pushover-Benachrichtigung senden, wenn eine andere Person als Jonas erkannt wurde
    if best_match_name and best_match_name != "Jonas":
        try:
            with open(img_path, "rb") as image_file:
                response = requests.post(
                    "https://api.pushover.net/1/messages.json",
                    data={
                        "token": "ar9c7y6i48rz1tg1jgf1n3q5rd2r1g",
                        "user": "ur8f5ekr9tca3cnrqdyihvxhrcmntp",
                        "message": f"Achtung: Person '{best_match_name}' erkannt!",
                        "priority": 2,
                        "sound": "Alarm",
                        "retry": 30,
                        "expire": 120
                    },
                    files={
                        "attachment": ("live.jpg", image_file, "image/jpeg")
                    }
                )
                print(f"Pushover-Benachrichtigung gesendet: {response.text}")
        except Exception as e:
            print(f"Fehler beim Senden der Pushover-Benachrichtigung: {e}")

    # Warte 10 Sekunden bis zur nächsten Überprüfung
    print("Warte 10 Sekunden bis zur nächsten Überprüfung...")
    time.sleep(10)


