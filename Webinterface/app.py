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
import numpy as np

app = Flask(__name__)

class FaceRecognition:
    """
    Schnelle, robuste Gesichtserkennung:
    - nutzt DeepFace 'Facenet512' + 'mtcnn'
    - cached Embeddings für bekannte Gesichter
    - liefert pro gefundenem Gesicht: Name, Confidence, Distance, Bounding-Box
    - thread-sicher
    """
    def __init__(self):
        self._deepface_ok = False
        self._lock = threading.RLock()
        self._model_name = 'Facenet512'
        self._detector = 'mtcnn'
        self._metric = 'cosine'  # für Facenet512 üblich
        self.known = []          # Liste: {'name','img_path','embedding':np.array}
        self._faces_json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Gesichtserkennung',
            'bekannte_gesichter.json'
        )
        self._faces_base_dir = os.path.dirname(self._faces_json_path)
        self._index_cache_path = os.path.join(self._faces_base_dir, 'bekannte_gesichter_index.json')
        self._init_deepface()
        self._load_known_faces(rebuild_index=False)  # lädt Cache, wenn möglich

    def _init_deepface(self):
        if self._deepface_ok:
            return
        try:
            from deepface import DeepFace  # noqa: F401
            self._deepface_ok = True
            print("✅ DeepFace verfügbar")
        except Exception as e:
            print(f"❌ DeepFace nicht verfügbar: {e}")
            self._deepface_ok = False

    def _ensure_deepface(self):
        if not self._deepface_ok:
            self._init_deepface()
        return self._deepface_ok

    def _represent_image(self, img_path):
        """Berechnet ein einzelnes Embedding (aligned) für ein Bildpfad."""
        from deepface import DeepFace
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=self._model_name,
            detector_backend=self._detector,
            enforce_detection=True,
            align=True,
            normalization='base'
        )
        # DeepFace.represent kann Liste zurückgeben; wir nehmen das erste Embedding
        if isinstance(reps, list) and len(reps) > 0 and 'embedding' in reps[0]:
            return np.asarray(reps[0]['embedding'], dtype='float32')
        # Fallback, falls ein einzelnes Dict kommt
        if isinstance(reps, dict) and 'embedding' in reps:
            return np.asarray(reps['embedding'], dtype='float32')
        raise ValueError("Konnte kein Embedding extrahieren.")

    def _save_index_cache(self):
        try:
            data = []
            for item in self.known:
                data.append({
                    "name": item['name'],
                    "img_path": os.path.basename(item['img_path']),
                    "embedding": item['embedding'].astype('float32').tolist()
                })
            with open(self._index_cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Konnte Index-Cache nicht speichern: {e}")

    def _load_index_cache(self):
        """Versucht, Embedding-Cache zu laden; gibt True bei Erfolg."""
        if not os.path.exists(self._index_cache_path):
            return False
        try:
            with open(self._index_cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.known = []
            for row in data:
                img_path = os.path.join(self._faces_base_dir, row['img_path'])
                emb = np.asarray(row['embedding'], dtype='float32')
                self.known.append({
                    'name': row['name'],
                    'img_path': img_path,
                    'embedding': emb
                })
            return True
        except Exception as e:
            print(f"⚠️ Konnte Index-Cache nicht laden: {e}")
            return False

    def _load_known_faces(self, rebuild_index=False):
        """Lädt bekannte Gesichter und (re)buildet Embedding-Index wenn nötig."""
        with self._lock:
            if not self._ensure_deepface():
                self.known = []
                return

            # JSON einlesen
            faces = []
            if os.path.exists(self._faces_json_path):
                with open(self._faces_json_path, 'r', encoding='utf-8') as f:
                    faces = json.load(f)
            else:
                self.known = []
                return

            # Falls kein rebuild angefordert: Cache versuchen
            if not rebuild_index and self._load_index_cache():
                # Prüfen, ob Anzahl/Namen/Dateien passen – wenn nicht, rebuild
                cache_names = sorted([(os.path.basename(k['img_path']), k['name']) for k in self.known])
                json_names = sorted([(f['Image'], f['Name']) for f in faces])
                if cache_names == json_names:
                    print(f"✅ Embedding-Cache geladen ({len(self.known)} Personen)")
                    return
                else:
                    print("ℹ️ Cache veraltet – baue neu")

            # Embeddings neu berechnen
            self.known = []
            for item in faces:
                name = item['Name']
                img_rel = item['Image']
                img_path = os.path.join(self._faces_base_dir, img_rel)
                if not os.path.exists(img_path):
                    print(f"⚠️ Bild fehlt: {img_path}")
                    continue
                try:
                    emb = self._represent_image(img_path)
                    self.known.append({'name': name, 'img_path': img_path, 'embedding': emb})
                except Exception as e:
                    print(f"⚠️ Embedding fehlgeschlagen für {name} ({img_rel}): {e}")

            print(f"✅ Embeddings erstellt: {len(self.known)} bekannte Personen")
            self._save_index_cache()

    def reload_known_faces(self):
        """Öffentliche Methode zum Reindex nach Änderungen (Add/Delete)."""
        self._load_known_faces(rebuild_index=True)

    def _cosine(self, a, b):
        # numerisch stabil
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return 1 - np.dot(a, b) / denom  # 0 = identisch, 2 = gegensätzlich

    def _threshold_for_model(self):
        """
        Erkennungsschwelle (cosine distance).
        Für Facenet512 sind ~0.3…0.4 üblich. Wir beginnen konservativ mit 0.35.
        """
        return 0.35

    def recognize_image(self, image_path):
        """
        Erkennt 0..n Gesichter in einem Bild.
        Rückgabe:
        {
          'faces': [
             {
               'recognized': True/False,
               'name': str|None,
               'confidence': float (0..1),
               'distance': float,
               'bbox': {'x':..,'y':..,'w':..,'h':..}
             }, ...
          ]
        }
        """
        with self._lock:
            if not self._ensure_deepface():
                return {'faces': []}

            from deepface import DeepFace

            if not os.path.exists(image_path):
                return {'faces': []}

            # 1) Gesichter extrahieren (liefert auch Bounding-Box)
            try:
                extracted = DeepFace.extract_faces(
                    img_path=image_path,
                    detector_backend=self._detector,
                    enforce_detection=True,
                    align=True
                )
            except Exception as e:
                print(f"⚠️ extract_faces Fehler: {e}")
                return {'faces': []}

            results = []
            if len(extracted) == 0:
                return {'faces': []}

            # 2) Für jedes gefundene Gesicht Embedding berechnen
            for face in extracted:
                # face['facial_area'] = {'x','y','w','h'}
                bbox = face.get('facial_area', {})
                # face['embedding'] ist nur vorhanden, wenn represent=True genutzt wurde.
                # Bei extract_faces ist i.d.R. nur das 'face' Bild enthalten -> also represent erneut:
                try:
                    emb = DeepFace.represent(
                        img_path=face['face'],
                        model_name=self._model_name,
                        detector_backend='skip',  # bereits aligned
                        enforce_detection=False,
                        align=False,
                        normalization='base'
                    )
                    if isinstance(emb, list):
                        emb = emb[0]['embedding']
                    elif isinstance(emb, dict):
                        emb = emb['embedding']
                    probe = np.asarray(emb, dtype='float32')
                except Exception as e:
                    print(f"⚠️ Embedding für Detected Face fehlgeschlagen: {e}")
                    continue

                # 3) Mit allen bekannten vergleichen
                best_name = None
                best_dist = 9e9
                for ref in self.known:
                    d = self._cosine(probe, ref['embedding'])
                    if d < best_dist:
                        best_dist = d
                        best_name = ref['name']

                thr = self._threshold_for_model()
                recognized = best_dist <= thr
                # Heuristische Confidence: mappe Distanz [0,thr] -> [1, ~0.5] und >thr -> <0.5
                if recognized:
                    confidence = max(0.0, min(1.0, 1.0 - (best_dist / (thr + 1e-9)) * 0.5))
                else:
                    confidence = max(0.0, 0.5 - min(0.5, (best_dist - thr)))  # fällt unter 0.5, je weiter weg

                results.append({
                    'recognized': bool(recognized),
                    'name': best_name if recognized else None,
                    'confidence': float(round(confidence, 3)),
                    'distance': float(round(best_dist, 4)),
                    'bbox': {
                        'x': int(bbox.get('x', 0)),
                        'y': int(bbox.get('y', 0)),
                        'w': int(bbox.get('w', 0)),
                        'h': int(bbox.get('h', 0)),
                    }
                })

            return {'faces': results}

    # Rückwärtskompatibel zu deinem bestehenden Endpoint
    def recognize_face_like_test1(self, image_path):
        res = self.recognize_image(image_path)
        # nimm bestes Gesicht (höchste Confidence)
        if not res['faces']:
            return {'recognized': False, 'name': None, 'confidence': 0}
        best = sorted(res['faces'], key=lambda f: f['confidence'], reverse=True)[0]
        return {
            'recognized': best['recognized'],
            'name': best['name'],
            'confidence': best['confidence']
        }

    # Hilfsfunktionen für Admin/Frontend
    def list_known(self):
        return [{'name': k['name'], 'image': os.path.basename(k['img_path'])} for k in self.known]

    def rebuild_index(self):
        self._load_known_faces(rebuild_index=True)
        return {'count': len(self.known)}

# Globale Instanz der Gesichtserkennung
face_recognition = FaceRecognition()

# Jeden Tag neuer Schlüssel
def generate_daily_secret_key():
    today = datetime.date.today().isoformat()
    random_bytes = os.urandom(16)  # Zufallswert für zusätzlichen Schutz
    return hashlib.sha256((today + str(random_bytes)).encode()).hexdigest()

app.secret_key = generate_daily_secret_key()

def get_db_connection():
    # Absoluter Pfad zur Datenbank im gleichen Verzeichnis wie app.py
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'homeshieldAI.db')
    return sqlite3.connect(db_path)

def check_login(username, password):
    connection = get_db_connection()
    cursor = connection.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    user = cursor.fetchone()
    connection.close()
    return user is not None

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
            if check_login(username, password):
                session["logged_in"] = True
                session["username"] = username
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
    
    # Kameras aus der Datenbank laden
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, name, ip_address, resolution FROM camera_settings ORDER BY id")
    cameras = cursor.fetchall()
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
    
    return render_template('dashboard.html', username=username, cameras=camera_list)

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
        
        # Temporäres Bild speichern
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_image_path = os.path.join(temp_dir, f'recognition_{timestamp}.jpg')
        
        with open(temp_image_path, 'wb') as f:
            f.write(response.content)
        
        # Gesichtserkennung durchführen
        result = face_recognition.recognize_face_like_test1(temp_image_path)
        
        # Temporäres Bild löschen
        try:
            os.remove(temp_image_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'recognized': result['recognized'],
            'name': result['name'],
            'confidence': result['confidence'],
            'camera': camera_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Fehler bei der Gesichtserkennung: {str(e)}'})

@app.route('/api/face_monitoring/status', methods=['GET'])
@login_required
def get_face_monitoring_status():
    """Gibt den Status der Gesichtserkennung zurück"""
    try:
        # Einfacher Status - immer verfügbar aber nicht kontinuierlich aktiv
        return jsonify({
            'success': True,
            'monitoring_active': False,  # Kein kontinuierliches Monitoring in der vereinfachten Version
            'cameras': {},
            'message': 'Gesichtserkennung verfügbar (manuell)'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Fehler beim Abrufen des Status: {str(e)}'})

@app.route('/api/face_monitoring/start', methods=['POST'])
@login_required
def start_global_face_monitoring():
    """Startet globale Gesichtserkennung (vereinfachte Version)"""
    try:
        return jsonify({
            'success': True,
            'message': 'Gesichtserkennung ist verfügbar - verwenden Sie die manuellen Erkennungsbuttons'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Fehler: {str(e)}'})

@app.route('/api/face_monitoring/stop', methods=['POST'])
@login_required
def stop_global_face_monitoring():
    """Stoppt globale Gesichtserkennung (vereinfachte Version)"""
    try:
        return jsonify({
            'success': True,
            'message': 'Kein kontinuierliches Monitoring aktiv'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Fehler: {str(e)}'})

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

if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0")