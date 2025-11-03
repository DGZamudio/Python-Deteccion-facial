from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from utils.fun import calcular_vectores_rostros, detectar_rostros, encontrar_rostros_similares
import numpy as np
import cv2

app = FastAPI()

@app.post("/reconocer-rostro")
async def reconocer_rostro(file: UploadFile = File(...)):
    contents = await file.read()

    # Convertir bytes → imagen OpenCV
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detectar rostros
    rostros = detectar_rostros(img)

    if not rostros:
        return JSONResponse(content={"error": "No se detectaron rostros."}, status_code=404)
    
    resultados = []
    for i, rostro in enumerate(rostros):
        # Calcular embedding
        embedding = calcular_vectores_rostros(rostro)

        # Buscar rostro similar en la base de datos
        persona = encontrar_rostros_similares(embedding)

        resultados.append({
            "id": i,
            "nombre": persona.get("nombre")
        })

    return JSONResponse(content={"resultados": resultados})
