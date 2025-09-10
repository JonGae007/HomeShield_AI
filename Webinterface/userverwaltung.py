import sqlite3
from getpass import getpass
import hashlib

connection = sqlite3.connect('Webinterface/homeshieldAI.db')
cursor = connection.cursor()
user = input("Enter username: ")
password = hashlib.sha256(getpass("Enter password: ").encode()).hexdigest()
create_table_query = '''INSERT INTO users (username, password) VALUES (?, ?)'''
cursor.execute(create_table_query, (user, password))
connection.commit()
connection.close()

