import cv2
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from database import conn

# Cargar archivo con python cv
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Cargar vectorización de rostros
clip_model = SentenceTransformer('clip-ViT-B-32', trust_remote_code=True)
clip_model._target_device = 'cpu' 
clip_model._first_module().processor.use_fast = True  # activar modo rápido

def detectar_rostros(imagen):
    # Poner imagen en blanco y negro
    gray_img = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

    # Detectar rostros
    rostros = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)
    )

    rostros_recortados = []
    
    # Por cada rostro detectado
    for x, y, w, h in rostros:
        cropped = imagen[y:y+h, x:x+w]
        rostro_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        rostros_recortados.append(rostro_pil)
    
    return rostros_recortados

def calcular_vectores_rostros(rostro_pil: Image.Image):
    # Calcular vectores del rostro
    embedding = clip_model.encode(rostro_pil, convert_to_tensor=True)
    
    return embedding.tolist()

def encontrar_rostros_similares(embedding):
    cur = conn.cursor()
    string_representation = "["+ ",".join(str(x) for x in embedding) +"]"
    
    #Buscar en la base de datos los rostros similares
    cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
    row = cur.fetchone()
    
    cur.close()

    if not row:
        return {"nombre": "Desconocido"}

    return {"nombre": row[1]}