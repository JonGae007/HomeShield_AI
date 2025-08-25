from flask import Flask, render_template, redirect, request, session, url_for, jsonify
import sqlite3
from functools import wraps
import os
import datetime
import hashlib
import socket
import requests
from urllib.parse import urlparse

app = Flask(__name__)

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

def check_camera_status(ip_address):
    """
    Überprüft ob eine Kamera online ist
    Versucht eine HTTP-Verbindung zur Kamera aufzubauen
    """
    if not ip_address:
        return False
    
    try:
        # Versuche HTTP-Verbindung zur Kamera (Timeout: 3 Sekunden)
        response = requests.get(f"http://{ip_address}", timeout=3)
        return response.status_code == 200
    except (requests.exceptions.RequestException, socket.error):
        try:
            # Fallback: Ping-Test über Socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((ip_address, 80))
            sock.close()
            return result == 0
        except:
            return False

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

    return render_template('login.html', error=error)

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
        # Status der Kamera überprüfen
        is_online = check_camera_status(camera[2])
        
        # Stream URL basierend auf Status setzen
        if is_online and camera[2]:
            stream_url = f"http://{camera[2]}/?action=stream"
        else:
            stream_url = "/static/pictures/static.png"
        
        camera_dict = {
            'id': camera[0],
            'name': camera[1],
            'ip_address': camera[2],
            'resolution': camera[3],
            'stream_url': stream_url,
            'status': 'online' if is_online else 'offline',
            'status_text': 'Online' if is_online else 'Offline'
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

@app.route('/account')
@login_required
def account():
    username = session.get('username', 'Guest')
    return render_template('account.html', username=username)

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
        
        # Überprüfen ob Kamera online ist
        if not check_camera_status(ip_address):
            connection.close()
            return jsonify({'success': False, 'error': 'Kamera ist offline'})
        
        # Foto von der Kamera abrufen
        try:
            response = requests.get(f"http://{ip_address}/?action=snapshot", timeout=10)
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

if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0")