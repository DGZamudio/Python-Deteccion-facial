from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
import cv2
import numpy as np
import os
import csv

app = FastAPI()

DATASET_PATH = "player_images"
NAMES_PATH = "players.csv"

# ---------------------------------------------------------
# Cargar diccionario desde el CSV (archivo_nombre → nombre)
# ---------------------------------------------------------
def cargar_diccionario():
    dic = {}
    if not os.path.exists(NAMES_PATH):
        raise FileNotFoundError(f"No se encontró el archivo {NAMES_PATH}")

    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ruta = row["ruta_imagen"]
            nombre_real = row["nombre_csv"]
            # extraer nombre del archivo sin extensión (para comparar)
            nombre_archivo = os.path.splitext(os.path.basename(ruta))[0]
            dic[nombre_archivo.lower()] = nombre_real
    return dic


# Cargar una sola vez al inicio
try:
    NOMBRES = cargar_diccionario()
    MODEL = DeepFace.build_model("Facenet")
    print(f"✅ Diccionario cargado con {len(NOMBRES)} jugadores.")
except Exception as e:
    print(f"⚠️ No se pudo cargar el diccionario: {e}")
    NOMBRES = {}


# ---------------------------------------------------------
# Función auxiliar: reconocimiento facial
# ---------------------------------------------------------
def reconocer_persona(img):
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=500, detail="No se encontró la carpeta 'player_images'")

    mejor_resultado = None
    mejor_distancia = 1.0  # cuanto menor, mejor

    for archivo in os.listdir(DATASET_PATH):
        if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path_referencia = os.path.join(DATASET_PATH, archivo)
        try:
            resultado = DeepFace.verify(
                img1_path=img,
                img2_path=path_referencia,
                enforce_detection=False,
                model_name="Facenet",
                model=MODEL
            )
            distancia = resultado["distance"]

            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_resultado = os.path.splitext(archivo)[0]
        except Exception:
            continue

    if mejor_resultado and mejor_distancia < 0.35:
        # Buscar el nombre real desde el CSV
        nombre_real = NOMBRES.get(mejor_resultado.lower(), mejor_resultado)
        return nombre_real, mejor_distancia
    else:
        return "Desconocido", None


# ---------------------------------------------------------
# Endpoint principal
# ---------------------------------------------------------
@app.post("/reconocer-rostro")
async def reconocer_rostro(file: UploadFile = File(...)):
    contents = await file.read()

    # Convertir bytes a imagen OpenCV
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    # Llamar al reconocedor
    nombre, distancia = reconocer_persona(img)

    return JSONResponse(content={
        "nombre": nombre,
        "distancia": distancia
    })
