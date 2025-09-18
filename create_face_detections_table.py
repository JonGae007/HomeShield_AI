import sqlite3
import os

def create_face_detections_table():
    """Erstellt die Tabelle für Gesichtserkennungs-Logs"""
    
    # Pfad zur Datenbank
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Webinterface', 'homeshieldAI.db')
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Erstelle face_detections Tabelle
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_known BOOLEAN NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera_id INTEGER,
                FOREIGN KEY (camera_id) REFERENCES camera_settings (id)
            )
        ''')
        
        # Index für bessere Performance bei Zeitbereichsabfragen
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detected_at ON face_detections (detected_at)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_name ON face_detections (name)
        ''')
        
        connection.commit()
        connection.close()
        
        print("✅ face_detections Tabelle erfolgreich erstellt")
        
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der Tabelle: {e}")

if __name__ == "__main__":
    create_face_detections_table()