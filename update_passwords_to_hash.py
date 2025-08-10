import sqlite3
import hashlib

def hash_passwords():
    connection = sqlite3.connect('homeshieldAI.db')
    cursor = connection.cursor()

    # Abrufen aller Benutzer und Passwörter
    cursor.execute("SELECT id, password FROM users")
    users = cursor.fetchall()

    for user_id, password in users:
        # Passwort in einen Hash umwandeln
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        # Aktualisieren des Passworts in der Datenbank
        cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))

    connection.commit()
    connection.close()

if __name__ == "__main__":
    hash_passwords()
    print("Passwörter wurden erfolgreich gehasht.")
