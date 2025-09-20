#!/usr/bin/env python3
"""
Erstellt die dashboard_settings Tabelle in der SQLite-Datenbank
für die Persistierung von Dashboard-Einstellungen
"""

import sqlite3
import os
from datetime import datetime

def create_dashboard_settings_table():
    """Erstellt die dashboard_settings Tabelle"""
    
    # Datenbankpfad
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Webinterface', 'homeshieldAI.db')
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Erstelle dashboard_settings Tabelle
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                setting_key TEXT NOT NULL,
                setting_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, setting_key)
            )
        ''')
        
        # Erstelle Trigger für automatisches Update der updated_at Spalte
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_dashboard_settings_timestamp 
            AFTER UPDATE ON dashboard_settings
            FOR EACH ROW
            BEGIN
                UPDATE dashboard_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        ''')
        
        # Erstelle Index für bessere Performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_dashboard_settings_user_key 
            ON dashboard_settings (user_id, setting_key)
        ''')
        
        connection.commit()
        print("✅ dashboard_settings Tabelle erfolgreich erstellt/aktualisiert")
        
        # Zeige aktuelle Tabellen
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📋 Verfügbare Tabellen: {[table[0] for table in tables]}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der Tabelle: {e}")
        return False

if __name__ == '__main__':
    print("🔧 Erstelle dashboard_settings Tabelle...")
    success = create_dashboard_settings_table()
    
    if success:
        print("✅ Setup abgeschlossen!")
    else:
        print("❌ Setup fehlgeschlagen!")