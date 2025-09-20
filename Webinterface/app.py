from flask import Flask, render_template, redirect, request, session, url_for, jsonify
import sqlite3
from functools import wraps
import os
import datetime
import hashlib
import requests
import json
import shutil
import uuid
import threading
import time
import cv2
import numpy as np
import face_recognition as fr
from PIL import Image
import io


app = Flask(__name__)

class FastFaceRecognition:
    """
    Schnelle Gesichtserkennung mit face-recognition library:
    - Erkennt bekannte und unbekannte Gesichter
    - Cached Encodings für Performance
    - Thread-sicher
    - Logging aller Erkennungen
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.known_faces = []  # Liste: {'name': str, 'encoding': np.array}
        self.detection_log = []  # Liste der letzten Erkennungen
        self._faces_json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Gesichtserkennung',
            'bekannte_gesichter.json'
        )
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Lädt bekannte Gesichter und erstellt Encodings"""
        with self._lock:
            self.known_faces = []
            
            if not os.path.exists(self._faces_json_path):
                print("❌ Keine bekannten Gesichter gefunden")
                return
            
            try:
                with open(self._faces_json_path, 'r', encoding='utf-8') as f:
                    faces_data = json.load(f)
                
                # Bilder sind im static/faces Ordner gespeichert
                faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'faces')
                
                for face_data in faces_data:
                    name = face_data['Name']
                    image_file = face_data['Image']
                    image_path = os.path.join(faces_dir, image_file)
                    
                    if os.path.exists(image_path):
                        # Lade Bild und erstelle Encoding
                        image = fr.load_image_file(image_path)
                        encodings = fr.face_encodings(image)
                        
                        if encodings:
                            self.known_faces.append({
                                'name': name,
                                'encoding': encodings[0]
                            })
                            print(f"✅ Gesicht geladen: {name}")
                        else:
                            print(f"⚠️ Kein Gesicht gefunden in {image_file}")
                    else:
                        print(f"⚠️ Bild nicht gefunden: {image_path}")
                
                print(f"✅ {len(self.known_faces)} bekannte Gesichter geladen")
                
            except Exception as e:
                print(f"❌ Fehler beim Laden der Gesichter: {e}")
    
    def detect_faces_in_image(self, image_data):
        """
        Erkennt alle Gesichter in einem Bild
        Rückgabe: {'faces': [{'name': str|'Unbekannt', 'confidence': float, 'location': tuple}]}
        """
        with self._lock:
            try:
                # Konvertiere Bilddaten
                if isinstance(image_data, bytes):
                    # Von bytes zu PIL Image zu numpy array
                    pil_image = Image.open(io.BytesIO(image_data))
                    image = np.array(pil_image)
                else:
                    # Direkt als numpy array verwenden
                    image = image_data
                
                # RGB konvertierung falls nötig
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Bereits RGB
                    rgb_image = image
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Gesichter finden
                face_locations = fr.face_locations(rgb_image, model="hog")  # "hog" ist schneller als "cnn"
                face_encodings = fr.face_encodings(rgb_image, face_locations)
                
                detected_faces = []
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Vergleiche mit bekannten Gesichtern
                    name = "Unbekannt"
                    confidence = 0.0
                    
                    if self.known_faces:
                        # Berechne Distanzen zu allen bekannten Gesichtern
                        known_encodings = [kf['encoding'] for kf in self.known_faces]
                        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                        face_distances = fr.face_distance(known_encodings, face_encoding)
                        
                        if True in matches:
                            # Finde beste Übereinstimmung
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_faces[best_match_index]['name']
                                confidence = 1.0 - face_distances[best_match_index]  # Je kleiner die Distanz, desto höher die Confidence
                    
                    detected_faces.append({
                        'name': name,
                        'confidence': float(confidence),
                        'location': face_location,  # (top, right, bottom, left)
                        'is_known': name != "Unbekannt"
                    })
                
                # Log the detection
                self._log_detection(detected_faces)
                
                return {'faces': detected_faces, 'total_faces': len(detected_faces)}
                
            except Exception as e:
                print(f"❌ Fehler bei Gesichtserkennung: {e}")
                return {'faces': [], 'total_faces': 0}
    
    def _log_detection(self, detected_faces):
        """Loggt Gesichtserkennungen in die Datenbank"""
        timestamp = datetime.datetime.now()
        
        for face in detected_faces:
            detection_entry = {
                'timestamp': timestamp.isoformat(),
                'name': face['name'],
                'confidence': face['confidence'],
                'is_known': face['is_known']
            }
            
            self.detection_log.append(detection_entry)
            
            # Speichere in Datenbank
            try:
                connection = get_db_connection()
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO face_detections (name, confidence, is_known, detected_at)
                    VALUES (?, ?, ?, ?)
                """, (face['name'], face['confidence'], face['is_known'], timestamp))
                connection.commit()
                connection.close()
            except Exception as e:
                print(f"❌ Fehler beim Speichern der Erkennung: {e}")
        
        # Halte Log-Liste klein (nur letzte 100 Einträge)
        if len(self.detection_log) > 100:
            self.detection_log = self.detection_log[-100:]
    
    def get_recent_detections(self, limit=50):
        """Gibt die letzten Erkennungen zurück"""
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("""
                SELECT name, confidence, is_known, detected_at 
                FROM face_detections 
                ORDER BY detected_at DESC 
                LIMIT ?
            """, (limit,))
            
            detections = []
            for row in cursor.fetchall():
                detections.append({
                    'name': row[0],
                    'confidence': row[1],
                    'is_known': bool(row[2]),
                    'detected_at': row[3]
                })
            
            connection.close()
            return detections
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der Erkennungen: {e}")
            return self.detection_log[-limit:] if self.detection_log else []
    
    def reload_known_faces(self):
        """Lädt bekannte Gesichter neu"""
        self._load_known_faces()

class FaceMonitoringService:
    """Service für kontinuierliche Gesichtserkennung im Hintergrund"""
    
    def __init__(self, face_recognizer, monitoring_interval=10):
        self.face_recognizer = face_recognizer
        self.monitoring_interval = monitoring_interval  # Sekunden zwischen Checks
        self.is_running = False
        self.monitoring_thread = None
        self.last_detection_time = None
        self.active_cameras = []
        self._lock = threading.RLock()
        self.auto_start_enabled = False  # Flag ob automatisch starten
        
    def load_settings_from_db(self, user_id):
        """Lädt Monitoring-Einstellungen aus der Datenbank für einen Benutzer"""
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            
            cursor.execute('''
                SELECT setting_key, setting_value 
                FROM dashboard_settings 
                WHERE user_id = ? AND setting_key IN ('monitoring_interval', 'monitoring_enabled')
            ''', (user_id,))
            
            settings = cursor.fetchall()
            connection.close()
            
            for key, value in settings:
                if key == 'monitoring_interval':
                    try:
                        import json
                        interval = json.loads(value) if value.isdigit() == False else int(value)
                        self.set_interval(interval)
                    except:
                        pass
                elif key == 'monitoring_enabled':
                    try:
                        import json
                        enabled = json.loads(value) if isinstance(value, str) and value.lower() in ['true', 'false'] else bool(value)
                        self.auto_start_enabled = enabled
                        if enabled and not self.is_running:
                            print("🔄 Auto-Start: Starte Monitoring basierend auf gespeicherten Einstellungen")
                            self.start_monitoring()
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Fehler beim Laden der Monitoring-Einstellungen: {e}")
    
    def save_settings_to_db(self, user_id):
        """Speichert aktuelle Monitoring-Einstellungen in die Datenbank"""
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            
            # Speichere Intervall
            cursor.execute('''
                INSERT OR REPLACE INTO dashboard_settings 
                (user_id, setting_key, setting_value, updated_at)
                VALUES (?, 'monitoring_interval', ?, CURRENT_TIMESTAMP)
            ''', (user_id, str(self.monitoring_interval)))
            
            # Speichere Status
            cursor.execute('''
                INSERT OR REPLACE INTO dashboard_settings 
                (user_id, setting_key, setting_value, updated_at)
                VALUES (?, 'monitoring_enabled', ?, CURRENT_TIMESTAMP)
            ''', (user_id, str(self.is_running)))
            
            connection.commit()
            connection.close()
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Monitoring-Einstellungen: {e}")
        
    def start_monitoring(self):
        """Startet das kontinuierliche Monitoring"""
        with self._lock:
            if self.is_running:
                print("⚠️ Monitoring läuft bereits")
                return
                
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            print(f"✅ Face Monitoring gestartet (Intervall: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stoppt das kontinuierliche Monitoring"""
        with self._lock:
            if not self.is_running:
                return
                
            self.is_running = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2)
            print("🛑 Face Monitoring gestoppt")
    
    def set_interval(self, seconds):
        """Ändert das Monitoring-Intervall"""
        with self._lock:
            self.monitoring_interval = max(5, min(300, seconds))  # 5-300 Sekunden
            print(f"⚙️ Monitoring-Intervall auf {self.monitoring_interval}s gesetzt")
    
    def get_status(self):
        """Gibt aktuellen Status zurück"""
        return {
            'is_running': self.is_running,
            'interval': self.monitoring_interval,
            'last_detection': self.last_detection_time.isoformat() if self.last_detection_time else None,
            'active_cameras': len(self.active_cameras)
        }
    
    def _get_active_cameras(self):
        """Holt aktive Kameras aus der Datenbank"""
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT id, ip_address, name FROM camera_settings ORDER BY id")
            cameras = cursor.fetchall()
            connection.close()
            
            active_cameras = []
            for camera in cameras:
                # Kurzer Ping-Test ob Kamera erreichbar ist
                try:
                    response = requests.get(f"http://{camera[1]}/?action=snapshot", timeout=2)
                    if response.status_code == 200:
                        active_cameras.append({
                            'id': camera[0],
                            'ip': camera[1], 
                            'name': camera[2]
                        })
                except:
                    pass  # Kamera offline oder nicht erreichbar
                    
            return active_cameras
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der Kameras: {e}")
            return []
    
    def _monitoring_loop(self):
        """Hauptschleife für kontinuierliches Monitoring"""
        print("🔄 Face Monitoring Loop gestartet")
        
        while self.is_running:
            try:
                # Aktive Kameras aktualisieren (alle 5 Zyklen)
                if len(self.active_cameras) == 0 or (hasattr(self, '_camera_check_counter') and self._camera_check_counter % 5 == 0):
                    self.active_cameras = self._get_active_cameras()
                    print(f"📹 {len(self.active_cameras)} aktive Kameras gefunden")
                
                if not hasattr(self, '_camera_check_counter'):
                    self._camera_check_counter = 0
                self._camera_check_counter += 1
                
                # Überwache jede aktive Kamera
                for camera in self.active_cameras:
                    if not self.is_running:
                        break
                        
                    try:
                        # Foto von Kamera aufnehmen
                        response = requests.get(f"http://{camera['ip']}/?action=snapshot", timeout=3)
                        
                        if response.status_code == 200:
                            # Gesichtserkennung durchführen
                            result = self.face_recognizer.detect_faces_in_image(response.content)
                            
                            if result['total_faces'] > 0:
                                self.last_detection_time = datetime.datetime.now()
                                
                                # Bekannte Gesichter gefunden?
                                known_faces = [f for f in result['faces'] if f['is_known']]
                                if known_faces:
                                    print(f"👤 {len(known_faces)} bekannte(s) Gesicht(er) erkannt auf {camera['name']}")
                                    for face in known_faces:
                                        print(f"   - {face['name']} (Confidence: {face['confidence']:.2f})")
                                
                                unknown_faces = [f for f in result['faces'] if not f['is_known']]
                                if unknown_faces:
                                    print(f"❓ {len(unknown_faces)} unbekannte(s) Gesicht(er) erkannt auf {camera['name']}")
                            
                    except Exception as e:
                        # Stille Fehler für bessere Performance
                        pass
                
                # Warte bis zum nächsten Zyklus
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"❌ Fehler im Monitoring Loop: {e}")
                time.sleep(5)  # Kurze Pause bei Fehlern
        
        print("🔚 Face Monitoring Loop beendet")

# Globale Instanzen
face_recognition = FastFaceRecognition()
face_monitoring = FaceMonitoringService(face_recognition, monitoring_interval=15)  # Alle 15 Sekunden

# Jeden Tag neuer Schlüssel
def generate_daily_secret_key():
    today = datetime.date.today().isoformat()
    random_bytes = os.urandom(16)  # Zufallswert für zusätzlichen Schutz
    return hashlib.sha256((today + str(random_bytes)).encode()).hexdigest()

app.secret_key = generate_daily_secret_key()

def get_db_connection():
    # Absoluter Pfad zur Datenbank im gleichen Verzeichnis wie app.py
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'homeshieldAI.db')
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row  # Enables column access by name
    return connection

def get_time_ago(timestamp_str):
    """Berechnet die Zeit seit einem Timestamp"""
    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"vor {diff.days} Tag{'en' if diff.days != 1 else ''}"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"vor {hours} Stunde{'n' if hours != 1 else ''}"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"vor {minutes} Minute{'n' if minutes != 1 else ''}"
        else:
            return "gerade eben"
    except:
        return "unbekannt"

def check_login(username, password):
    connection = get_db_connection()
    cursor = connection.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT id, username, password FROM users WHERE username=? AND password=?", (username, hashed_password))
    user = cursor.fetchone()
    connection.close()
    return user  # Gibt den ganzen User-Record zurück oder None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return redirect("/login")

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    message = request.args.get('message')
    
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            error = "Fehlende Eingaben."
        else:
            user = check_login(username, password)
            if user:
                session["logged_in"] = True
                session["username"] = username
                session["user_id"] = user[0]  # user[0] ist die ID
                print(f"🔐 Benutzer {username} angemeldet (ID: {user[0]})")
                return redirect("/dashboard")
            else:
                error = "Ungültige Anmeldedaten."

    return render_template('login.html', error=error, message=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect("/login")

@app.route('/dashboard')
@login_required
def dashboard():
    username = session.get('username', 'Guest')
    user_id = session.get('user_id')
    
    # Lade Dashboard-Einstellungen für den Benutzer
    global face_monitoring
    if user_id:
        face_monitoring.load_settings_from_db(user_id)
    
    # Kameras aus der Datenbank laden
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, name, ip_address, resolution FROM camera_settings ORDER BY id")
    cameras = cursor.fetchall()
    
    # Lade aktuelle Gesichtserkennungen (letzte 10)
    cursor.execute("""
        SELECT name, confidence, is_known, detected_at 
        FROM face_detections 
        ORDER BY detected_at DESC 
        LIMIT 10
    """)
    recent_detections = []
    for row in cursor.fetchall():
        recent_detections.append({
            'name': row[0],
            'confidence': round(row[1] * 100, 1),  # Als Prozent
            'is_known': bool(row[2]),
            'detected_at': row[3],
            'time_ago': get_time_ago(row[3])
        })
    
    connection.close()
    
    # Umwandlung in Dictionary für bessere Template-Verwendung
    camera_list = []
    for camera in cameras:
        stream_url = f"http://{camera[2]}/?action=stream" if camera[2] else "/static/pictures/static.png"
        
        camera_dict = {
            'id': camera[0],
            'name': camera[1],
            'ip_address': camera[2],
            'resolution': camera[3],
            'stream_url': stream_url
        }
        camera_list.append(camera_dict)
    
    return render_template('dashboard.html', 
                         username=username, 
                         cameras=camera_list,
                         recent_detections=recent_detections)

@app.route('/recordings')
@login_required
def recordings():
    username = session.get('username', 'Guest')
    
    # Captures aus der Datenbank laden
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, pfad, created_at FROM captures ORDER BY created_at DESC")
    captures_data = cursor.fetchall()
    connection.close()
    
    # Umwandlung in Dictionary für bessere Template-Verwendung
    captures = []
    for capture in captures_data:
        capture_dict = {
            'id': capture[0],
            'path': capture[1],
            'created_at': capture[2],
            'formatted_date': datetime.datetime.strptime(capture[2], '%Y-%m-%d %H:%M:%S').strftime('%d.%m.%Y %H:%M'),
            'filename': os.path.basename(capture[1])
        }
        captures.append(capture_dict)
    
    return render_template('recordings.html', username=username, captures=captures)

@app.route('/api/captures/<int:capture_id>', methods=['DELETE'])
@login_required
def delete_capture(capture_id):
    """API Endpoint um eine Aufnahme zu löschen"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Pfad der Aufnahme aus der Datenbank holen
        cursor.execute("SELECT pfad FROM captures WHERE id = ?", (capture_id,))
        capture = cursor.fetchone()
        
        if not capture:
            connection.close()
            return jsonify({'success': False, 'error': 'Aufnahme nicht gefunden'})
        
        # Datei löschen
        file_path = capture[0]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        
        if os.path.exists(full_path):
            os.remove(full_path)
        
        # Eintrag aus der Datenbank löschen
        cursor.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
        connection.commit()
        connection.close()
        
        return jsonify({'success': True, 'message': 'Aufnahme erfolgreich gelöscht'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Fehler beim Löschen: {str(e)}'})

@app.route('/faces')
@login_required
def faces():
    username = session.get('username', 'Guest')
    message = request.args.get('message')
    message_type = request.args.get('message_type', 'success')
    
    # Bekannte Gesichter aus JSON-Datei laden
    faces = []
    faces_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Gesichtserkennung', 'bekannte_gesichter.json')
    
    try:
        if os.path.exists(faces_json_path):
            with open(faces_json_path, 'r', encoding='utf-8') as f:
                known_faces_data = json.load(f)
            
            # Faces-Ordner erstellen falls nicht vorhanden
            static_faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'faces')
            os.makedirs(static_faces_dir, exist_ok=True)
            
            gesicht_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Gesichtserkennung')
            
            # Gesichter für Template aufbereiten
            for i, face_data in enumerate(known_faces_data):
                src_path = os.path.join(gesicht_dir, face_data['Image'])
                dst_path = os.path.join(static_faces_dir, face_data['Image'])
                
                # Bilder zu static/faces kopieren für Web-Zugriff
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                
                # Face-Objekt für Template erstellen
                face = {
                    'id': i + 1,
                    'name': face_data['Name'],
                    'image': face_data['Image'],
                    'added_date': datetime.datetime.now().strftime('%d.%m.%Y')
                }
                faces.append(face)
                
    except Exception as e:
        print(f"Fehler beim Laden der Gesichter: {e}")
    
    # Kameras für Modal laden
    cameras = []
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT id, name FROM camera_settings")
        cameras = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]
        connection.close()
    except Exception as e:
        print(f"Fehler beim Laden der Kameras: {e}")
    
    return render_template('faces.html', 
                         username=username, 
                         faces=faces, 
                         cameras=cameras,
                         message=message, 
                         message_type=message_type)

@app.route('/settings')
@login_required
def settings():
    username = session.get('username', 'Guest')
    
    # Kameras aus der Datenbank laden
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, name, ip_address, resolution FROM camera_settings ORDER BY id")
    cameras_data = cursor.fetchall()
    connection.close()
    
    # Umwandlung in Dictionary für JavaScript
    cameras = []
    for camera in cameras_data:
        camera_dict = {
            'id': camera[0],
            'name': camera[1],
            'ip_address': camera[2] or '',
            'resolution': camera[3] or '1920x1080'
        }
        cameras.append(camera_dict)
    
    return render_template('settings.html', username=username, cameras=cameras)

@app.route('/api/cameras', methods=['GET'])
@login_required
def get_cameras():
    """API Endpoint um alle Kameras zu laden"""
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, name, ip_address, resolution FROM camera_settings ORDER BY id")
    cameras_data = cursor.fetchall()
    connection.close()
    
    cameras = []
    for camera in cameras_data:
        camera_dict = {
            'id': camera[0],
            'name': camera[1],
            'ip_address': camera[2] or '',
            'resolution': camera[3] or '1920x1080'
        }
        cameras.append(camera_dict)
    
    return jsonify(cameras)

@app.route('/api/cameras/<int:camera_id>', methods=['PUT'])
@login_required
def update_camera(camera_id):
    """API Endpoint um eine Kamera zu aktualisieren"""
    data = request.get_json()
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Kamera aktualisieren
    cursor.execute("""
        UPDATE camera_settings 
        SET name = ?, ip_address = ?, resolution = ?
        WHERE id = ?
    """, (data.get('name'), data.get('ip_address'), data.get('resolution'), camera_id))
    
    connection.commit()
    connection.close()
    
    return jsonify({'success': True, 'message': 'Kamera erfolgreich aktualisiert'})

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
@login_required
def delete_camera(camera_id):
    """API Endpoint um eine Kamera zu löschen"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Kamera löschen
    cursor.execute("DELETE FROM camera_settings WHERE id = ?", (camera_id,))
    
    connection.commit()
    connection.close()
    
    return jsonify({'success': True, 'message': 'Kamera erfolgreich gelöscht'})

@app.route('/api/cameras', methods=['POST'])
@login_required
def create_camera():
    """API Endpoint um eine neue Kamera zu erstellen"""
    data = request.get_json()
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Neue Kamera erstellen
    cursor.execute("""
        INSERT INTO camera_settings (name, ip_address, resolution)
        VALUES (?, ?, ?)
    """, (data.get('name', 'Neue Kamera'), data.get('ip_address', ''), data.get('resolution', '1920x1080')))
    
    new_id = cursor.lastrowid
    connection.commit()
    connection.close()
    
    return jsonify({'success': True, 'message': 'Kamera erfolgreich erstellt', 'id': new_id})

@app.route('/api/cameras/<int:camera_id>/status')
@login_required
def check_camera_status(camera_id):
    """API Endpoint um den Status einer Kamera zu überprüfen"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT name, ip_address FROM camera_settings WHERE id = ?", (camera_id,))
        camera = cursor.fetchone()
        connection.close()
        
        if not camera:
            return jsonify({'success': False, 'error': 'Kamera nicht gefunden'})
        
        camera_name, ip_address = camera
        
        if not ip_address:
            return jsonify({'success': False, 'error': 'Kamera-IP nicht konfiguriert'})
        
        # Kurzer Test ob die Kamera erreichbar ist
        try:
            response = requests.get(f"http://{ip_address}/?action=snapshot", 
                                   timeout=3,
                                   headers={'User-Agent': 'HomeShieldAI/1.0'})
            response.raise_for_status()
            return jsonify({'success': True, 'status': 'online', 'message': 'Kamera ist erreichbar'})
        except requests.exceptions.RequestException as e:
            return jsonify({'success': False, 'status': 'offline', 'error': f'Kamera nicht erreichbar: {str(e)}'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unerwarteter Fehler: {str(e)}'})

@app.route('/logs')
@login_required
def logs():
    """Log-Seite für Gesichtserkennungen"""
    try:
        # Hole Parameter für Paginierung
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Hole Filter-Parameter
        name_filter = request.args.get('name', '').strip()
        known_filter = request.args.get('known', '')
        date_from = request.args.get('date_from', '')
        date_to = request.args.get('date_to', '')
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Baue WHERE Klausel basierend auf Filtern
        where_conditions = []
        params = []
        
        if name_filter:
            where_conditions.append("name LIKE ?")
            params.append(f"%{name_filter}%")
        
        if known_filter:
            if known_filter == 'known':
                where_conditions.append("is_known = 1")
            elif known_filter == 'unknown':
                where_conditions.append("is_known = 0")
        
        if date_from:
            where_conditions.append("date(detected_at) >= ?")
            params.append(date_from)
            
        if date_to:
            where_conditions.append("date(detected_at) <= ?")
            params.append(date_to)
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Gesamtanzahl für Paginierung
        count_query = f"SELECT COUNT(*) FROM face_detections{where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        # Berechne Offset
        offset = (page - 1) * per_page
        
        # Hole Erkennungen mit Paginierung
        query = f"""
            SELECT id, name, confidence, is_known, detected_at, camera_id
            FROM face_detections
            {where_clause}
            ORDER BY detected_at DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [per_page, offset])
        detections = cursor.fetchall()
        
        # Formatiere Daten
        formatted_detections = []
        for detection in detections:
            formatted_detections.append({
                'id': detection[0],
                'name': detection[1],
                'confidence': round(detection[2] * 100, 1),
                'is_known': detection[3],
                'detected_at': detection[4],
                'camera_id': detection[5],
                'time_ago': get_time_ago(detection[4])
            })
        
        connection.close()
        
        # Paginierungs-Informationen
        total_pages = (total_count + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages
        
        return render_template('logs.html', 
                             detections=formatted_detections,
                             page=page,
                             per_page=per_page,
                             total_count=total_count,
                             total_pages=total_pages,
                             has_prev=has_prev,
                             has_next=has_next,
                             name_filter=name_filter,
                             known_filter=known_filter,
                             date_from=date_from,
                             date_to=date_to)
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der Logs: {str(e)}")
        return render_template('logs.html', detections=[], error=str(e))

@app.route('/api/logs/<int:detection_id>', methods=['DELETE'])
@login_required
def delete_detection(detection_id):
    """API zum Löschen einer einzelnen Erkennung"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Prüfe ob Erkennung existiert
        cursor.execute("SELECT id FROM face_detections WHERE id = ?", (detection_id,))
        if not cursor.fetchone():
            connection.close()
            return jsonify({'success': False, 'error': 'Erkennung nicht gefunden'})
        
        # Lösche Erkennung
        cursor.execute("DELETE FROM face_detections WHERE id = ?", (detection_id,))
        connection.commit()
        connection.close()
        
        return jsonify({'success': True, 'message': 'Erkennung gelöscht'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/logs/bulk-delete', methods=['POST'])
@login_required
def bulk_delete_detections():
    """API zum Löschen mehrerer Erkennungen"""
    try:
        data = request.get_json()
        detection_ids = data.get('ids', [])
        
        if not detection_ids:
            return jsonify({'success': False, 'error': 'Keine IDs zum Löschen angegeben'})
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Erstelle Platzhalter für IN-Klausel
        placeholders = ','.join('?' * len(detection_ids))
        query = f"DELETE FROM face_detections WHERE id IN ({placeholders})"
        
        cursor.execute(query, detection_ids)
        deleted_count = cursor.rowcount
        connection.commit()
        connection.close()
        
        return jsonify({
            'success': True, 
            'message': f'{deleted_count} Erkennungen gelöscht'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/logs/clear-all', methods=['POST'])
@login_required
def clear_all_detections():
    """API zum Löschen aller Erkennungen (mit Bestätigung)"""
    try:
        data = request.get_json()
        confirm = data.get('confirm', False)
        
        if not confirm:
            return jsonify({'success': False, 'error': 'Bestätigung erforderlich'})
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Zähle vorher
        cursor.execute("SELECT COUNT(*) FROM face_detections")
        count_before = cursor.fetchone()[0]
        
        # Lösche alle
        cursor.execute("DELETE FROM face_detections")
        connection.commit()
        connection.close()
        
        return jsonify({
            'success': True, 
            'message': f'Alle {count_before} Erkennungen gelöscht'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/account')
@login_required
def account():
    username = session.get('username', 'Guest')
    error = request.args.get('error')
    success = request.args.get('success')
    return render_template('account.html', username=username, error=error, success=success)

@app.route('/capture_photo', methods=['POST'])
@login_required
def capture_photo():
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        camera_name = data.get('camera_name')
        
        if not camera_id:
            return jsonify({'success': False, 'error': 'Kamera-ID fehlt'})
        
        # Kamera-Daten aus der Datenbank laden
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT name, ip_address FROM camera_settings WHERE id = ?", (camera_id,))
        camera = cursor.fetchone()
        
        if not camera:
            connection.close()
            return jsonify({'success': False, 'error': 'Kamera nicht gefunden'})
        
        camera_name, ip_address = camera
        
        if not ip_address:
            connection.close()
            return jsonify({'success': False, 'error': 'Kamera-IP nicht konfiguriert'})
        
        # Foto von der Kamera abrufen (kürzerer Timeout)
        try:
            response = requests.get(f"http://{ip_address}/?action=snapshot", 
                                   timeout=5,
                                   headers={'User-Agent': 'HomeShieldAI/1.0'})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            connection.close()
            return jsonify({'success': False, 'error': f'Fehler beim Abrufen des Fotos: {str(e)}'})
        
        # Captures-Ordner erstellen falls er nicht existiert
        # Absoluter Pfad zum Webinterface-Verzeichnis
        base_dir = os.path.dirname(os.path.abspath(__file__))
        captures_dir = os.path.join(base_dir, 'static', 'pictures', 'captures')
        os.makedirs(captures_dir, exist_ok=True)
        
        # Dateiname generieren
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_name}_{timestamp}.jpg"
        filepath = os.path.join(captures_dir, filename)
        
        # Foto speichern
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        # Relativer Pfad für die Datenbank (für Web-Zugriff)
        relative_path = f"static/pictures/captures/{filename}"
        
        # Eintrag in Datenbank erstellen
        cursor.execute("INSERT INTO captures (pfad) VALUES (?)", (relative_path,))
        connection.commit()
        connection.close()
        
        return jsonify({'success': True, 'message': 'Foto erfolgreich aufgenommen', 'filename': filename})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unerwarteter Fehler: {str(e)}'})

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    try:
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        username = session.get('username')
        
        if not all([old_password, new_password, confirm_password]):
            return render_template('account.html', username=username, error='Alle Felder müssen ausgefüllt werden.')
        
        if new_password != confirm_password:
            return render_template('account.html', username=username, error='Die neuen Passwörter stimmen nicht überein.')
        
        if len(new_password) < 6:
            return render_template('account.html', username=username, error='Das neue Passwort muss mindestens 6 Zeichen lang sein.')
        
        # Altes Passwort überprüfen
        if not check_login(username, old_password):
            return render_template('account.html', username=username, error='Das aktuelle Passwort ist falsch.')
        
        # Neues Passwort hashen und in der Datenbank aktualisieren
        connection = get_db_connection()
        cursor = connection.cursor()
        hashed_new_password = hashlib.sha256(new_password.encode()).hexdigest()
        cursor.execute("UPDATE users SET password=? WHERE username=?", (hashed_new_password, username))
        connection.commit()
        connection.close()
        
        return render_template('account.html', username=username, success='Passwort erfolgreich geändert.')
        
    except Exception as e:
        return render_template('account.html', username=username, error=f'Fehler beim Ändern des Passworts: {str(e)}')

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    try:
        password = request.form.get('password')
        username = session.get('username')
        
        if not password:
            return render_template('account.html', username=username, error='Passwort zur Bestätigung erforderlich.')
        
        # Passwort überprüfen
        if not check_login(username, password):
            return render_template('account.html', username=username, error='Falsches Passwort.')
        
        # Benutzer aus der Datenbank löschen
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM users WHERE username=?", (username,))
        connection.commit()
        connection.close()
        
        # Session beenden
        session.clear()
        
        return redirect(url_for('login', message='Account erfolgreich gelöscht.'))
        
    except Exception as e:
        return render_template('account.html', username=username, error=f'Fehler beim Löschen des Accounts: {str(e)}')

@app.route('/add_face', methods=['POST'])
@login_required
def add_face():
    try:
        name = request.form.get('name', '').strip()
        
        if not name:
            return redirect(url_for('faces', message='Name ist erforderlich', message_type='error'))
        
        # Gesichter JSON-Datei Pfad
        faces_json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'Gesichtserkennung', 
            'bekannte_gesichter.json'
        )
        gesicht_dir = os.path.dirname(faces_json_path)
        
        # Bestehende Gesichter laden
        known_faces = []
        if os.path.exists(faces_json_path):
            with open(faces_json_path, 'r', encoding='utf-8') as f:
                known_faces = json.load(f)
        
        # Prüfen ob Name bereits existiert
        for face in known_faces:
            if face['Name'].lower() == name.lower():
                return redirect(url_for('faces', message=f'Ein Gesicht mit dem Namen "{name}" existiert bereits', message_type='error'))
        
        image_filename = None
        
        # Bild-Upload verarbeiten
        if 'image' in request.files and request.files['image'].filename:
            file = request.files['image']
            if file and file.filename:
                file_ext = os.path.splitext(file.filename)[1].lower()
                if file_ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                    return redirect(url_for('faces', message='Nur JPG, PNG und GIF Dateien erlaubt', message_type='error'))
                
                image_filename = f"{uuid.uuid4()}{file_ext}"
                image_path = os.path.join(gesicht_dir, image_filename)
                file.save(image_path)
        
        # Kamerafoto verarbeiten
        elif 'camera_id' in request.form and request.form.get('camera_id'):
            camera_id = request.form.get('camera_id')
            
            # Kamera-URL aus Datenbank holen
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT name, ip_address FROM camera_settings WHERE id = ?", (camera_id,))
            camera = cursor.fetchone()
            connection.close()
            
            if not camera:
                return redirect(url_for('faces', message='Kamera nicht gefunden', message_type='error'))
            
            camera_name, camera_ip = camera
            
            try:
                # Foto von Kamera aufnehmen
                response = requests.get(f"http://{camera_ip}/?action=snapshot", timeout=10)
                response.raise_for_status()
                
                # Bild speichern
                image_filename = f"{uuid.uuid4()}.jpg"
                image_path = os.path.join(gesicht_dir, image_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                    
            except Exception as e:
                return redirect(url_for('faces', message=f'Fehler beim Aufnehmen des Fotos: {str(e)}', message_type='error'))
        
        else:
            return redirect(url_for('faces', message='Bild oder Kamerafoto ist erforderlich', message_type='error'))
        
        if not image_filename:
            return redirect(url_for('faces', message='Fehler beim Verarbeiten des Bildes', message_type='error'))
        
        # Neues Gesicht zur Liste hinzufügen
        new_face = {
            'Name': name,
            'Image': image_filename
        }
        known_faces.append(new_face)
        
        # JSON-Datei aktualisieren
        with open(faces_json_path, 'w', encoding='utf-8') as f:
            json.dump(known_faces, f, ensure_ascii=False, indent=2)
        
        # Bild zu static/faces kopieren
        static_faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'faces')
        os.makedirs(static_faces_dir, exist_ok=True)
        src_path = os.path.join(gesicht_dir, image_filename)
        dst_path = os.path.join(static_faces_dir, image_filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        
        # FaceRecognition neu laden
        face_recognition.reload_known_faces()
        
        return redirect(url_for('faces', message=f'Gesicht "{name}" wurde erfolgreich hinzugefügt', message_type='success'))
        
    except Exception as e:
        print(f"Fehler beim Hinzufügen des Gesichts: {e}")
        return redirect(url_for('faces', message=f'Fehler beim Hinzufügen des Gesichts: {str(e)}', message_type='error'))

@app.route('/delete_face/<int:face_id>', methods=['POST'])
@login_required  
def delete_face(face_id):
    try:
        # Gesichter JSON-Datei laden
        faces_json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'Gesichtserkennung', 
            'bekannte_gesichter.json'
        )
        gesicht_dir = os.path.dirname(faces_json_path)
        
        if not os.path.exists(faces_json_path):
            return jsonify({'success': False, 'message': 'Keine Gesichter gefunden'})
        
        with open(faces_json_path, 'r', encoding='utf-8') as f:
            known_faces = json.load(f)
        
        # Gesicht löschen (face_id ist 1-basiert)
        if 1 <= face_id <= len(known_faces):
            face_to_delete = known_faces[face_id - 1]
            face_name = face_to_delete['Name']
            image_filename = face_to_delete['Image']
            
            # Aus Liste entfernen
            known_faces.pop(face_id - 1)
            
            # JSON-Datei aktualisieren
            with open(faces_json_path, 'w', encoding='utf-8') as f:
                json.dump(known_faces, f, ensure_ascii=False, indent=2)
            
            # Bilddateien löschen
            image_paths = [
                os.path.join(gesicht_dir, image_filename),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'faces', image_filename)
            ]
            
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            # FaceRecognition neu laden
            face_recognition.reload_known_faces()
            
            return jsonify({'success': True, 'message': f'Gesicht "{face_name}" wurde gelöscht'})
        else:
            return jsonify({'success': False, 'message': 'Ungültige Gesicht-ID'})
            
    except Exception as e:
        print(f"Fehler beim Löschen des Gesichts: {e}")
        return jsonify({'success': False, 'message': f'Fehler beim Löschen: {str(e)}'})

@app.route('/api/camera/<int:camera_id>/recognize_face', methods=['POST'])
@login_required
def recognize_face_from_camera(camera_id):
    """Nimmt ein Foto von der Kamera auf und führt Gesichtserkennung durch"""
    try:
        # Kamera-Details aus der Datenbank holen
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT name, ip_address FROM camera_settings WHERE id=?", (camera_id,))
        camera_data = cursor.fetchone()
        connection.close()
        
        if not camera_data:
            return jsonify({'success': False, 'error': 'Kamera nicht gefunden'})
        
        camera_name, ip_address = camera_data
        
        # Foto von der Kamera aufnehmen
        image_url = f"http://{ip_address}/?action=snapshot"
        response = requests.get(image_url, timeout=5)
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': 'Konnte kein Foto von der Kamera aufnehmen'})
        
        # Gesichtserkennung durchführen
        result = face_recognition.detect_faces_in_image(response.content)
        
        return jsonify({
            'success': True,
            'faces': result['faces'],
            'total_faces': result['total_faces'],
            'camera': camera_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Fehler bei der Gesichtserkennung: {str(e)}'})

@app.route('/api/face_detections/recent', methods=['GET'])
@login_required
def get_recent_face_detections():
    """API für aktuelle Gesichtserkennungen"""
    try:
        limit = request.args.get('limit', 10, type=int)
        detections = face_recognition.get_recent_detections(limit)
        
        # Formatiere für Frontend
        formatted_detections = []
        for detection in detections:
            formatted_detections.append({
                'name': detection['name'],
                'confidence': round(detection['confidence'] * 100, 1),
                'is_known': detection['is_known'],
                'detected_at': detection['detected_at'],
                'time_ago': get_time_ago(detection['detected_at'])
            })
        
        return jsonify({'success': True, 'detections': formatted_detections})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/face_detections/statistics', methods=['GET'])
@login_required 
def get_face_detection_statistics():
    """API für Gesichtserkennungs-Statistiken"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Statistiken für heute
        cursor.execute("""
            SELECT 
                COUNT(*) as total_today,
                SUM(CASE WHEN is_known = 1 THEN 1 ELSE 0 END) as known_today,
                COUNT(DISTINCT name) as unique_today
            FROM face_detections 
            WHERE date(detected_at) = date('now')
        """)
        
        today_stats = cursor.fetchone()
        
        # Statistiken für diese Woche
        cursor.execute("""
            SELECT 
                COUNT(*) as total_week,
                SUM(CASE WHEN is_known = 1 THEN 1 ELSE 0 END) as known_week,
                COUNT(DISTINCT name) as unique_week
            FROM face_detections 
            WHERE datetime(detected_at) >= datetime('now', '-7 days')
        """)
        
        week_stats = cursor.fetchone()
        
        connection.close()
        
        statistics = {
            'today': {
                'total': today_stats[0] or 0,
                'known': today_stats[1] or 0,
                'unknown': (today_stats[0] or 0) - (today_stats[1] or 0),
                'unique': today_stats[2] or 0
            },
            'week': {
                'total': week_stats[0] or 0,
                'known': week_stats[1] or 0,
                'unknown': (week_stats[0] or 0) - (week_stats[1] or 0),
                'unique': week_stats[2] or 0
            }
        }
        
        return jsonify({'success': True, 'statistics': statistics})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/face_monitoring/start_continuous', methods=['POST'])
@login_required
def start_continuous_monitoring():
    """Startet kontinuierliche Gesichtserkennung für alle Kameras"""
    try:
        # TODO: Implementiere kontinuierliche Überwachung
        return jsonify({
            'success': True, 
            'message': 'Kontinuierliche Gesichtserkennung gestartet',
            'status': 'running'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Face monitoring API routes removed - face recognition functionality disabled

@app.route('/api/cameras/<int:camera_id>/status', methods=['GET'])
@login_required
def get_camera_status(camera_id):
    """API Endpoint um den Status einer einzelnen Kamera zu überprüfen"""
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT ip_address FROM camera_settings WHERE id = ?", (camera_id,))
    camera = cursor.fetchone()
    connection.close()
    
    if not camera or not camera[0]:
        return jsonify({'status': 'offline', 'status_text': 'Offline'})
    
    # Einfacher Online-Check
    try:
        response = requests.get(f"http://{camera[0]}/?action=snapshot", timeout=3)
        is_online = response.status_code == 200
    except:
        is_online = False
    
    return jsonify({
        'status': 'online' if is_online else 'offline',
        'status_text': 'Online' if is_online else 'Offline'
    })

@app.route('/api/cameras/status', methods=['GET'])
@login_required
def get_all_cameras_status():
    """API Endpoint um den Status aller Kameras zu überprüfen"""
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, ip_address FROM camera_settings ORDER BY id")
    cameras = cursor.fetchall()
    connection.close()
    
    status_list = []
    for camera in cameras:
        try:
            response = requests.get(f"http://{camera[1]}/?action=snapshot", timeout=3)
            is_online = response.status_code == 200
        except:
            is_online = False
            
        status_list.append({
            'id': camera[0],
            'status': 'online' if is_online else 'offline',
            'status_text': 'Online' if is_online else 'Offline'
        })
    
    return jsonify(status_list)

@app.route('/api/face_monitoring/status', methods=['GET'])
@login_required
def face_monitoring_status():
    """API Endpoint für Face Monitoring Status"""
    global face_recognition, face_monitoring
    
    if face_recognition is None:
        return jsonify({
            'status': 'disabled',
            'status_text': 'Gesichtserkennung nicht aktiv',
            'known_faces': 0,
            'recent_detections': 0,
            'monitoring': {'is_running': False, 'interval': 0}
        })
    
    # Aktuelle Statistiken
    known_faces_count = len(face_recognition.known_faces)
    
    # Zähle Erkennungen der letzten 24 Stunden
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM face_detections 
        WHERE detected_at >= datetime('now', '-24 hours')
    """)
    recent_detections = cursor.fetchone()['count']
    connection.close()
    
    # Monitoring Status
    monitoring_status = face_monitoring.get_status()
    
    return jsonify({
        'status': 'active',
        'status_text': 'Gesichtserkennung aktiv',
        'known_faces': known_faces_count,
        'recent_detections': recent_detections,
        'monitoring': monitoring_status
    })

@app.route('/api/face_monitoring/start', methods=['POST'])
@login_required
def start_face_monitoring():
    """Startet kontinuierliches Face Monitoring"""
    global face_monitoring
    
    try:
        user_id = session.get('user_id')
        
        # Optional: Intervall aus Request holen
        data = request.get_json() or {}
        interval = data.get('interval', 15)  # Standard: 15 Sekunden
        
        face_monitoring.set_interval(interval)
        face_monitoring.start_monitoring()
        
        # Einstellungen in Datenbank speichern
        face_monitoring.save_settings_to_db(user_id)
        
        return jsonify({
            'success': True, 
            'message': f'Face Monitoring gestartet (alle {interval}s)',
            'status': face_monitoring.get_status()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/face_monitoring/stop', methods=['POST'])
@login_required
def stop_face_monitoring():
    """Stoppt kontinuierliches Face Monitoring"""
    global face_monitoring
    
    try:
        user_id = session.get('user_id')
        
        face_monitoring.stop_monitoring()
        
        # Einstellungen in Datenbank speichern
        face_monitoring.save_settings_to_db(user_id)
        
        return jsonify({
            'success': True, 
            'message': 'Face Monitoring gestoppt',
            'status': face_monitoring.get_status()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/face_monitoring/interval', methods=['POST'])
@login_required
def set_monitoring_interval():
    """Ändert das Monitoring-Intervall"""
    global face_monitoring
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        interval = int(data.get('interval', 15))
        
        if interval < 5 or interval > 300:
            return jsonify({'success': False, 'error': 'Intervall muss zwischen 5 und 300 Sekunden liegen'})
        
        face_monitoring.set_interval(interval)
        
        # Einstellungen in Datenbank speichern
        face_monitoring.save_settings_to_db(user_id)
        
        return jsonify({
            'success': True, 
            'message': f'Intervall auf {interval}s gesetzt',
            'status': face_monitoring.get_status()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Dashboard Settings API
@app.route('/api/dashboard/settings', methods=['GET'])
@login_required
def get_dashboard_settings():
    """Lädt Dashboard-Einstellungen für den aktuellen Benutzer"""
    try:
        user_id = session.get('user_id')
        print(f"📋 Loading dashboard settings for user_id: {user_id}")
        
        if user_id is None:
            return jsonify({'success': False, 'error': 'Keine Benutzer-ID in Session gefunden'})
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Lade alle Einstellungen für den Benutzer
        cursor.execute('''
            SELECT setting_key, setting_value 
            FROM dashboard_settings 
            WHERE user_id = ?
        ''', (user_id,))
        
        settings_rows = cursor.fetchall()
        connection.close()
        
        print(f"📊 Found {len(settings_rows)} settings in database for user {user_id}")
        
        # Konvertiere zu Dictionary
        settings = {}
        for key, value in settings_rows:
            # Versuche JSON zu parsen, falls nicht möglich als String
            try:
                import json
                settings[key] = json.loads(value)
            except:
                settings[key] = value
            print(f"  - {key}: {settings[key]}")
        
        # Standard-Einstellungen falls keine vorhanden
        default_settings = {
            'monitoring_interval': 15,
            'monitoring_enabled': False,
            'auto_refresh': True,
            'refresh_interval': 30
        }
        
        # Merge mit Defaults
        for key, default_value in default_settings.items():
            if key not in settings:
                settings[key] = default_value
        
        return jsonify({
            'success': True,
            'settings': settings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dashboard/settings', methods=['POST'])
@login_required
def save_dashboard_settings():
    """Speichert Dashboard-Einstellungen für den aktuellen Benutzer"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        if not data or 'settings' not in data:
            return jsonify({'success': False, 'error': 'Keine Einstellungen übermittelt'})
        
        settings = data['settings']
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Speichere jede Einstellung
        for key, value in settings.items():
            # Konvertiere zu JSON String
            import json
            value_str = json.dumps(value) if not isinstance(value, str) else str(value)
            
            # INSERT OR REPLACE für jede Einstellung
            cursor.execute('''
                INSERT OR REPLACE INTO dashboard_settings 
                (user_id, setting_key, setting_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, key, value_str))
        
        connection.commit()
        connection.close()
        
        return jsonify({
            'success': True,
            'message': f'{len(settings)} Einstellungen gespeichert'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dashboard/setting/<setting_key>', methods=['POST'])
@login_required
def save_single_dashboard_setting(setting_key):
    """Speichert eine einzelne Dashboard-Einstellung"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        print(f"💾 Saving setting {setting_key} for user_id: {user_id}, data: {data}")
        
        if user_id is None:
            return jsonify({'success': False, 'error': 'Keine Benutzer-ID in Session gefunden'})
        
        if not data or 'value' not in data:
            return jsonify({'success': False, 'error': 'Kein Wert übermittelt'})
        
        value = data['value']
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Konvertiere zu JSON String falls nötig
        import json
        value_str = json.dumps(value) if not isinstance(value, str) else str(value)
        
        # INSERT OR REPLACE
        cursor.execute('''
            INSERT OR REPLACE INTO dashboard_settings 
            (user_id, setting_key, setting_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, setting_key, value_str))
        
        connection.commit()
        print(f"✅ Setting {setting_key} = {value} saved to database for user {user_id}")
        
        # Verifikation: Prüfe ob es wirklich gespeichert wurde
        cursor.execute('SELECT setting_value FROM dashboard_settings WHERE user_id = ? AND setting_key = ?', 
                      (user_id, setting_key))
        saved_value = cursor.fetchone()
        if saved_value:
            print(f"✅ Verification: {setting_key} is now {saved_value[0]} in database")
        else:
            print(f"❌ Verification failed: {setting_key} not found in database")
        
        connection.close()
        
        return jsonify({
            'success': True,
            'message': f'Einstellung {setting_key} gespeichert'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0")