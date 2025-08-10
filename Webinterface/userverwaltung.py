import sqlite3
from getpass import getpass

# Verbindung mit check_same_thread=False erstellen
connection = sqlite3.connect('homeshieldAI.db', check_same_thread=False)
cursor = connection.cursor()

user = input("Enter username: ")
password = getpass("Enter password: ")
create_table_query = '''INSERT INTO users (username, password) VALUES (?, ?)'''
cursor.execute(create_table_query, (user, password))

connection.commit()
connection.close()

