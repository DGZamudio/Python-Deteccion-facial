import os
from utils.fun import calcular_vectores_rostros
from database import conn
from PIL import Image

def almacenar_rostros():
    cur = conn.cursor()

    for file_name in os.listdir('./data/player_images'):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Saltar archivos no válidos

        # Abrir la imagen
        img_path = os.path.join('./data/player_images', file_name)
        img = Image.open(img_path)

        # Calcular embedding
        embedding = calcular_vectores_rostros(img)

        # Convertir a string formato vector PostgreSQL
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        nombre_jugador = file_name.replace("_", " ").split(".")[0]

        # Guardar en la base de datos
        cur.execute("""
            INSERT INTO pictures (nombre, embedding)
            VALUES (%s, %s)
        """, (nombre_jugador, embedding_str))

    conn.commit()
    cur.close()
    print("✅ Rostros almacenados correctamente.")

    
almacenar_rostros()