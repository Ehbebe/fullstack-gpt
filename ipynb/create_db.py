import sqlite3

conn = sqlite3.connect("movies.sqlite", isolation_level=None)
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS directors;")
c.execute(
    """
CREATE TABLE directors (
id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
name TEXT,
gender INTEGER,
uid INTEGER
);
"""
)
c.execute(
    """INSERT INTO directors VALUES
(1, 'James Cameron', 2, 1),
(2, 'Gore Verbinski', 2, 2),
(3, 'Sam Mendes', 2, 3),
(4, 'Christopher Nolan', 2, 4),
(5, 'Andrew Stanton', 2, 5);
"""
)

c.execute("DROP TABLE IF EXISTS movies;")
c.execute(
    """
CREATE TABLE movies (
id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
original_title TEXT,
budget TEXT,
popularity [REAL],
director_id [INTEGER],
gross_revenue [REAL]
)
"""
)
c.execute(
    """INSERT INTO movies VALUES
(1, 'Avatar', 237000000, 150.437577, 1, 2787965087),
(2, 'Pirates of the Caribbean: At World''s End', 300000000, 139.082615, 2, 961000000),
(3, 'Spectre', 245000000, 107.376788, 3, 880674609),
(4, 'The Dark Knight Rises', 250000000, 112.31295, 4, 1084939099),
(5, 'John Carter', 260000000, 43.926995, 5, 284139100);
"""
)
c.close()
