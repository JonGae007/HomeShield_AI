import cv2
import os
import json

# path = "" # Wird später initialisiert
gesichter_daten = []
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, "bekannte_gesichter.json")

# Versuche, vorhandene Daten zu laden
try:
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            gesichter_daten = json.load(f)
            if not isinstance(gesichter_daten, list):
                print(f"Warnung: Inhalt von {json_file_path} ist keine Liste. Beginne mit leeren Daten.")
                gesichter_daten = []
    else:
        print(f"Info: {json_file_path} nicht gefunden. Es wird eine neue Datei erstellt.")
except json.JSONDecodeError:
    print(f"Warnung: {json_file_path} enthält ungültiges JSON. Beginne mit leeren Daten.")
    gesichter_daten = []
except Exception as e:
    print(f"Fehler beim Laden von {json_file_path}: {e}. Beginne mit leeren Daten.")
    gesichter_daten = []

path = "" # Initialisiere path hier

while not path == "end":
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        break
    path = input("Name (oder 'end' zum Beenden): ")
    if path == "end":
        del(camera) # Kamera freigeben, bevor die Schleife verlassen wird
        break
    if not path: # Leere Eingabe überspringen
        print("Bitte einen Namen eingeben.")
        del(camera)
        continue

    return_value, image = camera.read()
    if not return_value:
        print("Fehler: Bild konnte nicht von der Kamera aufgenommen werden.")
        del(camera)
        continue

    bild_dateiname = path + ".jpg"
    bild_speicherpfad = os.path.join(script_dir, bild_dateiname)
    cv2.imwrite(bild_speicherpfad, image)

    # Prüfen, ob der Name bereits existiert und ggf. aktualisieren oder neu hinzufügen
    eintrag_gefunden = False
    for eintrag in gesichter_daten:
        if eintrag["Name"] == path:
            eintrag["Image"] = bild_dateiname  # Bildname aktualisieren
            print(f"Eintrag für '{path}' aktualisiert mit Bild '{bild_dateiname}'.")
            eintrag_gefunden = True
            break
    
    if not eintrag_gefunden:
        data = {"Name": path, "Image": bild_dateiname}
        gesichter_daten.append(data)
        print(f"'{path}' gespeichert als '{bild_dateiname}'.")
    
    del(camera) # Kamera nach Gebrauch freigeben

# Speichere die gesammelten Daten in der JSON-Datei
try:
    with open(json_file_path, 'w') as f:
        json.dump(gesichter_daten, f, indent=4)
    print(f"Alle Daten wurden in '{json_file_path}' gespeichert.")
except IOError as e:
    print(f"Fehler beim Schreiben der JSON-Datei {json_file_path}: {e}")
except Exception as e:
    print(f"Ein unerwarteter Fehler ist beim Speichern der JSON-Datei aufgetreten: {e}")