import psycopg2
import os
# from dotenv import load_dotenv

# load_dotenv()

DATABASE_CONN = os.getenv('DATABASE_CONN')

conn = psycopg2.connect(DATABASE_CONN)

def crear_tabla_si_no_existe():
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS pictures (
            id SERIAL PRIMARY KEY,
            nombre TEXT,
            embedding vector(512)
        );
    """)
    conn.commit()
    cur.close()
    
def traer_datos():
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM pictures
    """)
    row = cur.fetchone()
    print(row)
    conn.commit()
    cur.close()
    
def eliminar_registros():
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM pictures
        WHERE true;
    """)
    conn.commit()
    cur.close()