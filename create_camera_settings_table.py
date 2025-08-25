#!/usr/bin/env python3
"""
Skript zum Erstellen einer Tabelle für Kamera-Einstellungen in der HomeShieldAI Datenbank.
Dieses Skript erstellt eine neue Tabelle 'camera_settings' falls sie noch nicht existiert.
"""

import sqlite3
import os
import sys

def create_camera_settings_table():
    """
    Erstellt eine neue Tabelle für Kamera-Einstellungen in der SQLite-Datenbank.
    """
    # Pfad zur Datenbank
    db_path = 'homeshieldAI.db'
    
    # Prüfen ob Datenbank existiert
    if not os.path.exists(db_path):
        print(f"Fehler: Datenbank '{db_path}' nicht gefunden!")
        print("Stelle sicher, dass du das Skript im richtigen Verzeichnis ausführst.")
        return False
    
    try:
        # Verbindung zur Datenbank herstellen
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # SQL-Befehl zum Erstellen der Tabelle
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS camera_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL DEFAULT 'Kamera',
            ip_address VARCHAR(50) DEFAULT '',
            resolution VARCHAR(20) DEFAULT '1920x1080',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        '''
        
        # Tabelle erstellen
        cursor.execute(create_table_query)
        
        # Trigger für automatische Aktualisierung von updated_at erstellen
        trigger_query = '''
        CREATE TRIGGER IF NOT EXISTS update_camera_settings_timestamp 
        AFTER UPDATE ON camera_settings
        FOR EACH ROW
        BEGIN
            UPDATE camera_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        '''
        
        cursor.execute(trigger_query)
        
        # Änderungen speichern
        connection.commit()
        
        print("✓ Tabelle 'camera_settings' erfolgreich erstellt!")
        print("\nTabellen-Schema:")
        print("- id: Eindeutige ID (AUTO_INCREMENT)")
        print("- name: Kamera-Name")
        print("- ip_address: IP-Adresse der Kamera")
        print("- resolution: Auflösung (z.B. 1920x1080)")
        print("- created_at: Erstellungszeitpunkt")
        print("- updated_at: Letzte Aktualisierung")
        
        # Aktuelle Daten anzeigen
        cursor.execute("SELECT COUNT(*) FROM camera_settings")
        count = cursor.fetchone()[0]
        print(f"\n✓ Tabelle erstellt - Aktuell {count} Kamera-Einträge vorhanden")
        
        return True
        
    except sqlite3.Error as e:
        print(f"Datenbankfehler: {e}")
        return False
        
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")
        return False
        
    finally:
        if connection:
            connection.close()

def show_existing_tables():
    """
    Zeigt alle existierenden Tabellen in der Datenbank an.
    """
    db_path = 'homeshieldAI.db'
    
    if not os.path.exists(db_path):
        print(f"Datenbank '{db_path}' nicht gefunden!")
        return
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Existierende Tabellen in der Datenbank:")
        for table in tables:
            print(f"- {table[0]}")
            
    except sqlite3.Error as e:
        print(f"Fehler beim Anzeigen der Tabellen: {e}")
    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    print("HomeShieldAI - Kamera-Einstellungen Tabelle Setup")
    print("=" * 50)
    
    # Zeige existierende Tabellen
    print("\n1. Überprüfe existierende Tabellen:")
    show_existing_tables()
    
    print("\n2. Erstelle camera_settings Tabelle:")
    success = create_camera_settings_table()
    
    if success:
        print("\n✓ Setup erfolgreich abgeschlossen!")
        print("\nDie Tabelle 'camera_settings' kann jetzt für die Speicherung")
        print("von Kamera-Einstellungen im Webinterface verwendet werden.")
    else:
        print("\n✗ Setup fehlgeschlagen!")
        sys.exit(1)
