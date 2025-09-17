#!/usr/bin/env python3
"""
Vereinfachte Version der HomeShield AI App ohne komplexe DeepFace-Konfiguration
"""

import os
import sys
import json
import time
import threading
import sqlite3
import datetime
import uuid
import requests as session_requests
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory, flash
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Konfiguration
UPLOAD_FOLDER = 'static/faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_db_connection():
    """Erstellt Datenbankverbindung"""
    connection = sqlite3.connect('homeshieldAI.db')
    connection.row_factory = sqlite3.Row
    return connection

def allowed_file(filename):
    """Prüft ob Dateiendung erlaubt ist"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class FaceRecognition:
    def __init__(self):
        self.known_names = []
        self.known_faces = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.camera_results = {}
        self.load_known_faces()

    def load_known_faces(self):
        """Lädt bekannte Gesichter aus der JSON-Datei"""
        try:
            faces_json_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'Gesichtserkennung', 
                'bekannte_gesichter.json'
            )
            
            if os.path.exists(faces_json_path):
                with open(faces_json_path, 'r', encoding='utf-8') as f:
                    gesichter_daten = json.load(f)
                    
                self.known_names = []
                self.known_faces = []
                
                faces_dir = os.path.dirname(faces_json_path)
                for eintrag in gesichter_daten:
                    self.known_names.append(eintrag["Name"])
                    img_path = os.path.join(faces_dir, eintrag["Image"])
                    self.known_faces.append(img_path)
                    
                print(f"Bekannte Gesichter geladen: {self.known_names}")
        except Exception as e:
            print(f"Fehler beim Laden der Gesichter: {e}")

    def recognize_face_deepface_like_test1(self, image_path):
        """Gesichtserkennung wie in Test1.py - mit DeepFace"""
        try:
            from deepface import DeepFace
            
            # Überprüfe ob die Input-Datei existiert
            if not os.path.exists(image_path):
                print(f"Input-Bild nicht gefunden: {image_path}")
                return {'recognized': False, 'name': None, 'confidence': 0}

            best_match_name = None
            min_distance = float('inf')
            
            # Verwende die gleichen Einstellungen wie Test1.py
            detector_to_use = 'mtcnn'
            model_to_use = 'Facenet512'

            print(f"Versuche Gesichter mit Detektor '{detector_to_use}' und Modell '{model_to_use}' zu vergleichen.")

            for i, known_img_path in enumerate(self.known_faces):
                try:
                    print(f"Vergleiche mit {self.known_names[i]} ({os.path.basename(known_img_path)})...")
                    
                    result = DeepFace.verify(
                        img1_path=image_path, 
                        img2_path=known_img_path, 
                        detector_backend=detector_to_use,
                        model_name=model_to_use,
                        enforce_detection=True
                    ) 
                    
                    print(f"  Vergleich von '{os.path.basename(image_path)}' mit '{os.path.basename(known_img_path)}' ({self.known_names[i]}):")
                    print(f"  Ergebnis: {result}")

                    if result["verified"] and result["distance"] < min_distance:
                        min_distance = result["distance"]
                        best_match_name = self.known_names[i]
                        
                except Exception as e:
                    print(f"Fehler beim Vergleichen mit {self.known_names[i]} ({os.path.basename(known_img_path)}): {e}")

            if best_match_name:
                print(f"Das Gesicht in {os.path.basename(image_path)} gehört am ehesten zu {best_match_name} (Distanz: {min_distance:.4f})")
                return {
                    'recognized': True,
                    'name': best_match_name,
                    'confidence': max(0, 1 - min_distance)
                }
            else:
                print(f"Keine Übereinstimmung für {os.path.basename(image_path)} gefunden mit den aktuellen Einstellungen.")
                return {'recognized': False, 'name': None, 'confidence': 0}
                
        except ImportError:
            print("DeepFace nicht installiert. Verwende OpenCV Fallback.")
            return self.recognize_face_opencv_fallback(image_path)
        except Exception as e:
            print(f"Fehler bei der Gesichtserkennung: {e}")
            return {'recognized': False, 'name': None, 'confidence': 0}

    def recognize_face_opencv_fallback(self, image_path):
        """Fallback Gesichtserkennung mit OpenCV"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            img = cv2.imread(image_path)
            if img is None:
                return {'recognized': False, 'name': None, 'confidence': 0}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                print(f"🔍 {len(faces)} Gesicht(er) erkannt - OpenCV Fallback")
                return {'recognized': False, 'name': None, 'confidence': 0.3}
            else:
                return {'recognized': False, 'name': None, 'confidence': 0}
                
        except Exception as e:
            print(f"Fehler bei OpenCV Gesichtserkennung: {e}")
            return {'recognized': False, 'name': None, 'confidence': 0}

    def detect_faces_in_image(self, image_path):
        """Erkennt Anzahl der Gesichter im Bild und klassifiziert sie"""
        try:
            if not os.path.exists(image_path):
                print(f"Bilddatei nicht gefunden: {image_path}")
                return {'status': 'no_face', 'faces_count': 0, 'known_faces': [], 'unknown_faces': 0}
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Bild konnte nicht geladen werden: {image_path}")
                return {'status': 'no_face', 'faces_count': 0, 'known_faces': [], 'unknown_faces': 0}
            
            # Gesichtserkennung mit OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                return {'status': 'no_face', 'faces_count': 0, 'known_faces': [], 'unknown_faces': 0}
            
            print(f"🔍 {len(faces)} Gesicht(er) erkannt in {os.path.basename(image_path)}")
            
            # Für jedes erkannte Gesicht prüfen ob bekannt
            known_faces = []
            unknown_count = 0
            
            # DeepFace Erkennung wie in Test1.py
            if self.known_faces:
                try:
                    recognition_result = self.recognize_face_deepface_like_test1(image_path)
                    
                    if recognition_result['recognized'] and recognition_result['name']:
                        known_faces.append({
                            'name': recognition_result['name'],
                            'confidence': recognition_result['confidence']
                        })
                        if len(faces) > 1:
                            unknown_count = len(faces) - 1
                        print(f"✅ Bekanntes Gesicht gefunden: {recognition_result['name']} (Vertrauen: {recognition_result['confidence']:.3f})")
                    else:
                        unknown_count = len(faces)
                        print(f"❓ Unbekannte Gesichter: {unknown_count}")
                except Exception as recognition_error:
                    print(f"Erkennungsfehler: {recognition_error}")
                    unknown_count = len(faces)
            else:
                print("Keine bekannten Gesichter zum Vergleich verfügbar")
                unknown_count = len(faces)
            
            status = 'known_face' if known_faces else 'unknown_face'
            
            result = {
                'status': status,
                'faces_count': len(faces),
                'known_faces': known_faces,
                'unknown_faces': unknown_count
            }
            
            print(f"📊 Ergebnis: Status={status}, Gesamt={len(faces)}, Bekannt={len(known_faces)}, Unbekannt={unknown_count}")
            
            return result
            
        except Exception as e:
            print(f"Fehler bei der Gesichtserkennung in {image_path}: {e}")
            return {'status': 'error', 'faces_count': 0, 'known_faces': [], 'unknown_faces': 0}

    def start_monitoring(self):
        """Startet kontinuierliche Überwachung aller Kameras"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("🔍 Gesichtserkennung-Überwachung gestartet")

    def stop_monitoring(self):
        """Stoppt kontinuierliche Überwachung"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("⏹️ Gesichtserkennung-Überwachung gestoppt")

    def _monitoring_loop(self):
        """Hauptschleife für kontinuierliche Überwachung"""
        while self.monitoring_active:
            try:
                # Alle aktiven Kameras laden
                connection = get_db_connection()
                cursor = connection.cursor()
                cursor.execute("SELECT id, name, ip_address FROM camera_settings WHERE ip_address != '' AND ip_address IS NOT NULL")
                cameras = cursor.fetchall()
                connection.close()
                
                # Jede Kamera prüfen
                for camera_id, camera_name, ip_address in cameras:
                    if not self.monitoring_active:
                        break
                        
                    try:
                        print(f"📸 Prüfe Kamera {camera_name} ({ip_address})")
                        
                        # Foto von Kamera aufnehmen
                        response = session_requests.get(
                            f"http://{ip_address}/?action=snapshot", 
                            timeout=5,
                            headers={'User-Agent': 'HomeShieldAI/1.0'}
                        )
                        
                        if response.status_code == 200 and len(response.content) > 1000:
                            # Temporäres Bild speichern
                            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            temp_image_path = os.path.join(temp_dir, f'monitoring_cam{camera_id}_{timestamp}.jpg')
                            
                            try:
                                with open(temp_image_path, 'wb') as f:
                                    f.write(response.content)
                                
                                print(f"📁 Bild gespeichert: {os.path.basename(temp_image_path)} ({len(response.content)} bytes)")
                                
                                # Gesichtserkennung durchführen
                                result = self.detect_faces_in_image(temp_image_path)
                                result['camera_id'] = camera_id
                                result['camera_name'] = camera_name
                                result['timestamp'] = datetime.datetime.now().isoformat()
                                result['online'] = True
                                
                                # Ergebnis speichern
                                self.camera_results[camera_id] = result
                                
                                print(f"✅ Kamera {camera_name}: {result['status']} - {result['faces_count']} Gesicht(er)")
                                
                            finally:
                                # Temporäres Bild löschen
                                try:
                                    if os.path.exists(temp_image_path):
                                        os.remove(temp_image_path)
                                        print(f"🗑️ Temp-Bild gelöscht: {os.path.basename(temp_image_path)}")
                                except Exception as cleanup_error:
                                    print(f"Warnung: Temp-Datei konnte nicht gelöscht werden: {cleanup_error}")
                                
                        else:
                            print(f"❌ Kamera {camera_name}: Offline oder ungültige Antwort")
                            self.camera_results[camera_id] = {
                                'camera_id': camera_id,
                                'camera_name': camera_name,
                                'status': 'offline',
                                'online': False,
                                'timestamp': datetime.datetime.now().isoformat(),
                                'faces_count': 0,
                                'known_faces': [],
                                'unknown_faces': 0
                            }
                            
                    except Exception as e:
                        print(f"❌ Fehler bei Kamera {camera_name}: {e}")
                        self.camera_results[camera_id] = {
                            'camera_id': camera_id,
                            'camera_name': camera_name,
                            'status': 'error',
                            'online': False,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'error': str(e),
                            'faces_count': 0,
                            'known_faces': [],
                            'unknown_faces': 0
                        }
                
                # 5 Sekunden warten vor nächstem Durchlauf
                time.sleep(5)
                
            except Exception as e:
                print(f"Fehler in Überwachungsschleife: {e}")
                time.sleep(10)

    def get_camera_status(self, camera_id=None):
        """Gibt den aktuellen Status einer oder aller Kameras zurück"""
        if camera_id:
            return self.camera_results.get(camera_id, {
                'camera_id': camera_id,
                'status': 'no_data',
                'online': False,
                'timestamp': datetime.datetime.now().isoformat()
            })
        else:
            return self.camera_results

# Globale Instanz der Gesichtserkennung
face_recognition = FaceRecognition()

# Initialisierung direkt beim Start
face_recognition.start_monitoring()

# Login erforderlich Decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        connection = get_db_connection()
        user = connection.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        connection.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash('Ungültiger Benutzername oder Passwort')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    connection = get_db_connection()
    cameras = connection.execute('SELECT * FROM camera_settings').fetchall()
    connection.close()
    return render_template('dashboard.html', cameras=cameras)

@app.route('/faces')
@login_required
def faces():
    return render_template('faces.html')

@app.route('/settings')
@login_required
def settings():
    connection = get_db_connection()
    cameras = connection.execute('SELECT * FROM camera_settings').fetchall()
    connection.close()
    return render_template('settings.html', cameras=cameras)

@app.route('/recordings')
@login_required
def recordings():
    return render_template('recordings.html')

@app.route('/account')
@login_required
def account():
    return render_template('account.html')

# API Routes für Face Recognition
@app.route('/api/face_monitoring/status')
@login_required
def api_face_monitoring_status():
    """API Endpoint für Face Monitoring Status"""
    return jsonify(face_recognition.get_camera_status())

@app.route('/api/cameras/<int:camera_id>/status')
@login_required
def api_camera_status(camera_id):
    """API Endpoint für einzelnen Kamera-Status"""
    return jsonify(face_recognition.get_camera_status(camera_id))

if __name__ == '__main__':
    print("🚀 HomeShield AI - Vereinfachte Version")
    print("📡 Server startet auf http://localhost:80")
    app.run(debug=True, host='0.0.0.0', port=80)