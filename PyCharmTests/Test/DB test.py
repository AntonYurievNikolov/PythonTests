import sqlite3
conn = sqlite3.connect('runs.db')
with conn:
    c = conn.cursor()
    c.execute('CREATE TABLE Person(id INTEGER PRIMARY KEY, name TEXT)')
    c.execute('CREATE TABLE Location(id INTEGER PRIMARY KEY, place TEXT)')
    c.execute('''CREATE TABLE Run(id INTEGER PRIMARY KEY,
    pid INTEGER,
    lid INTEGER,
    distance NUMBER,
    FOREIGN KEY(pid) REFERENCES Person(id),
    FOREIGN KEY(lid) REFERENCES Location(id)
    )
    ''')
    c.execute("INSERT INTO Person VALUES(0, 'Matt')")
    c.execute("INSERT INTO Location VALUES(0, 'Squaw Peak')")
    c.execute("INSERT INTO Run VALUES(0, 0, 0, 52)")
    c.execute("INSERT INTO Run VALUES(1, 0, 0, 50)")

q = 'SELECT * FROM Person WHERE name IS NOT NULL'