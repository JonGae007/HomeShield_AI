from flask import Flask, render_template, redirect, request
import sqlite3

app = Flask(__name__)

def get_db_connection():
    # Verbindung bei jeder Anfrage neu erstellen
    return sqlite3.connect('homeshieldAI.db')

def check_login(username, password):
    # Benutzer suchen
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    connection.close()
    return user is not None  # True, wenn gefunden


@app.route('/')
def home():
    return redirect("/login", code=302)

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
                return redirect("/dashboard", code=302)
            else:
                error = "Ungültige Anmeldedaten."

    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/recordings')
def recordings():
    return render_template('recordings.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0")