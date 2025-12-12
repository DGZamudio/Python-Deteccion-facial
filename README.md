🏀 Face Recognition API – Identificación de Jugadores

API en FastAPI para reconocer rostros de jugadores de baloncesto a partir de una imagen.
El sistema analiza la foto, genera un embedding facial y lo compara con los datos almacenados en una base de datos para identificar al jugador.

🚀 ¿Qué hace esta API?

Recibe una imagen vía POST.

Detecta uno o varios rostros.

Genera los embeddings.

Compara cada embedding con los guardados en la base de datos.

Devuelve el nombre del jugador más similar.

🔧 Tecnologías usadas

Python

FastAPI

OpenCV

Numpy

face_recognition (dlib)

PostgreSQL

psycopg2

📸 Endpoint principal
POST /reconocer-rostro

Envías una imagen. La API devuelve una lista con los rostros detectados y el nombre del jugador identificado.

Ejemplo de respuesta:

{
  "resultados": [
    {
      "id": 0,
      "nombre": "LeBron James"
    }
  ]
}

▶️ Ejecutar el proyecto

Instalar dependencias:

pip install -r requirements.txt


Ejecutar la API:

uvicorn app.main:app --reload
