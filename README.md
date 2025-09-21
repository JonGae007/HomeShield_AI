
HomeShield AI
=============

Kurze Beschreibung
-------------------
HomeShield AI ist eine einfache Webanwendung zur Überwachung mit Gesichtserkennung und Kameraverwaltung. Die Anwendung stellt ein Webinterface bereit, in dem Kameras konfiguriert, erkannte Gesichter verwaltet und Aufnahmen/Logs eingesehen werden können. Sie ist als leichtes, lokal ausführbares Projekt konzipiert und speichert Einstellungen sowie Erkennungsdaten in einer SQLite-Datenbank.

Features
--------
- Webinterface zum Ansehen von Dashboards, Kameras, Erkennungen und Logs
- Verwaltung erkannter Gesichter (Name, Zuordnung zur Kamera)
- Speicherung von Einstellungen und Erkennungsdaten in `homeshieldAI.db` (SQLite)
- Statische Oberfläche mit einfachen CSS/JS-Ressourcen

Technologien
-----------
- Python 3
- Flask (einfacher Webserver, siehe `Webinterface/app.py`)
- SQLite (lokale Datenbank: `Webinterface/homeshieldAI.db`)
- HTML/CSS/JavaScript für das Frontend (im Ordner `Webinterface/static` und `Webinterface/templates`)

Projektstruktur
----------------
Wichtige Dateien und Ordner:

- `Webinterface/`
	- `app.py`: Flask-Applikation (Entry-Point für die Weboberfläche)
	- `homeshieldAI.db`: SQLite-Datenbank mit Kamera- und Erkennungsdaten
	- `static/`: Statische Dateien (CSS, JS, Bilder, Icons)
	- `templates/`: HTML-Templates für die Seiten (Login, Dashboard, Faces, Logs, Settings)
- `requirements.txt`: Python-Abhängigkeiten
- `README.md`: Dieses Dokument

Installation
------------
Voraussetzungen:

- Python 3.8+ auf macOS/Linux/Windows
- `pip` zum Installieren von Abhängigkeiten

Schritte:

1. Repository klonen (falls noch nicht geschehen):

```bash
git clone https://github.com/JonGae007/HomeShield_AI.git
cd HomeShield_AI/Webinterface
```

2. Virtuelle Umgebung erstellen (empfohlen):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Abhängigkeiten installieren:

```bash
pip install -r ../requirements.txt
```

Hinweis: Das `requirements.txt` liegt im Projektstamm; bei Bedarf in das `Webinterface`-Verzeichnis kopieren oder den Pfad anpassen.

Nutzung / Starten
-----------------
Wechsle in das `Webinterface`-Verzeichnis und starte die Flask-App:

```bash
cd Webinterface
python3 app.py
```

Öffne dann im Browser `http://127.0.0.1:5000/` (oder die im Terminal angezeigte Adresse).

Datenbank
--------
Die Anwendung verwendet `homeshieldAI.db` (SQLite) zur Ablage von:

- Kamerakonfigurationen (`camera_settings`)
- Erkannten Gesichtern (`face_detections` oder ähnliche Tabellen)
- Logs und weitere Einstellungen

Du kannst die Datenbank mit `sqlite3` direkt untersuchen, z. B.:

```bash
sqlite3 Webinterface/homeshieldAI.db "SELECT name, ip_address FROM camera_settings;"
```

Entwicklung und Anpassung
-------------------------
- Änderungen am Webinterface werden in `Webinterface/app.py`, `Webinterface/templates` und `Webinterface/static` vorgenommen.
- Für die Einbindung einer echten Gesichtserkennung (z. B. OpenCV + face_recognition) lässt sich `app.py` erweitern — aktuell ist die Struktur auf Speicherung und Anzeige ausgelegt.
- Wenn du neue Python-Pakete benötigst, ergänze `requirements.txt` und installiere sie in der virtuellen Umgebung.

Sicherheit und Datenschutz
-------------------------
- Die Anwendung speichert Bild- und Gesichtsdaten lokal in der SQLite-Datenbank.
- Stelle sicher, dass Zugriff und Speicherung den lokalen Datenschutzvorgaben entsprechen.

Autor
-----
Jonas Gärtner und Raik Remmers (Repository: `JonGae007/HomeShield_AI`)

Standardzugang
---------------
Der Standardbenutzer für das Webinterface ist `admin` mit dem Passwort `password`. Bitte ändere dieses Passwort nach dem ersten Login.
