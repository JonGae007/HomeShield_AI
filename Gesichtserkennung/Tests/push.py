import requests
import os  # Hinzugefügt

# Pfad zum Bild relativ zum Skriptverzeichnis erstellen
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "live_view.jpg")

with open(image_path, "rb") as image_file:  # Geänderter Pfad zur Bilddatei
    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": "ar9c7y6i48rz1tg1jgf1n3q5rd2r1g",  # Token beibehalten
            "user": "ur8f5ekr9tca3cnrqdyihvxhrcmntp",  # User-Key beibehalten
            "message": "Person erkannt!",  # Nachricht gemäß Bild aktualisiert
            "priority": 2,  # Priorität auf 2 geändert
            "sound": "Alarm",  # Sound "sad" gemäß Bild hinzugefügt
            "retry": 30,  # Erneut versuchen alle 60 Sekunden
            "expire": 120 # Läuft nach 1 Stunde ab
        },
        files={
            "attachment": ("live.jpg", image_file, "image/jpeg")  # Dateianhang beibehalten
        }
    )
print(response.text)