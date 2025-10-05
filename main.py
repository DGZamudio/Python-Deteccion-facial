from fastapi import FastAPI, File, Response, UploadFile, HTTPException
import cv2
import numpy as np
import uuid
import os
import base64

app = FastAPI()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

os.makedirs("resultados", exist_ok=True)

@app.post('/detectar-rostros')
async def detectar_rostros(file: UploadFile = File(...)):
    try:
        contents = await file.read()
    except:
        HTTPException(400, "Error al decodificar la imagten, imagen no cargada o invalida")
        
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR) 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,   
        minNeighbors=3,    
        minSize=(30, 30) 
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    filename = f"resultados/{uuid.uuid4()}.jpg"
    cv2.imwrite(filename, img)
    
    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
