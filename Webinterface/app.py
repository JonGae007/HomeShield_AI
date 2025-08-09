from flask import Flask, render_template, redirect
import sqlite3

app = Flask(__name__)

db = sqlite3.connect('users.db')
cursor = db.cursor()

@app.route('/')
def home():
    return redirect("/dashboard", code=302)

@app.route('/login')
def login():
    return render_template('login.html')

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
