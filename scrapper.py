import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import csv
import random
import time
from PIL import Image

# === CONFIGURACIÓN ===
csv_file = "players.csv"
images_dir = "player_images"
os.makedirs(images_dir, exist_ok=True)

# === HEADERS PARA EVITAR BLOQUEOS ===
HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
]

# === FUNCIONES ===
def mejorar_resolucion(path, tamaño=(224, 224)):
    """
    Redimensiona la imagen a tamaño fijo.
    ArcFace y Facenet funcionan bien con 112x112 o 224x224.
    """
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(tamaño, Image.LANCZOS)  # LANCZOS = buena calidad
        img.save(path)
        return True
    except Exception as e:
        print(f"⚠️ No se pudo mejorar la imagen {path}: {e}")
        return False

def get_wikipedia_image(player_name: str):
    """Intenta obtener la imagen del jugador desde Wikipedia."""
    query = requests.utils.quote(f"{player_name} basketball player")
    url = f"https://en.wikipedia.org/wiki/Special:Search?search={query}"
    headers = random.choice(HEADERS_LIST)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Buscar imagen dentro del infobox (panel lateral)
        infobox = soup.find("table", class_="infobox")
        if infobox:
            img_tag = infobox.find("img")
            if img_tag and img_tag.get("src"):
                img_url = "https:" + img_tag["src"]
                return img_url
    except Exception as e:
        print(f"⚠️ Error en Wikipedia para {player_name}: {e}")

    return None


def get_google_image(player_name: str):
    """Fallback: busca imagen del jugador en Google (primer resultado real)."""
    query = requests.utils.quote(f"{player_name} basketball player face portrait")
    url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = random.choice(HEADERS_LIST)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.select("img")

        # Ignorar logos y data URLs
        for img in img_tags:
            src = img.get("src")
            if src and src.startswith("http") and "googlelogo" not in src:
                return src
    except Exception as e:
        print(f"⚠️ Error en Google para {player_name}: {e}")

    return None


def download_image(url, save_path):
    """Descarga la imagen si el link es válido."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Imagen guardada: {save_path}")
            return True
        else:
            print(f"⚠️ Error {response.status_code} al descargar {url}")
    except Exception as e:
        print(f"⚠️ No se pudo descargar {url}: {e}")
    return False


# === SCRIPT PRINCIPAL ===
if __name__ == "__main__":
    # Cargar base de datos de tu amigo
    df = pd.read_csv("./yizak/Project-Programming-Fundamentals/data/shot_logs.csv")
    players = df["player_name"].dropna().unique().tolist()

    # Si ya existe un CSV previo, cargarlo para no repetir jugadores
    done = set()
    if os.path.exists(csv_file):
        prev = pd.read_csv(csv_file)
        done = set(prev["nombre_csv"].tolist())

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.stat(csv_file).st_size == 0:
            writer.writerow(["nombre_csv", "ruta_imagen"])

        for i, name in enumerate(players, start=1):
            if name in done:
                print(f"⏩ Saltando {name}, ya descargado.")
                continue

            print(f"🔍 [{i}/{len(players)}] Buscando imagen para: {name}")
            img_url = get_wikipedia_image(name)

            if not img_url:
                print(f"⚠️ No se encontró en Wikipedia, buscando en Google…")
                img_url = get_google_image(name)

            if not img_url:
                print(f"❌ No se encontró imagen para {name}")
                continue

            filename = f"{name.replace(' ', '_')}.jpg"
            save_path = os.path.join(images_dir, filename)

            if download_image(img_url, save_path):
                mejorar_resolucion(save_path)
                writer.writerow([name, save_path])
                done.add(name)

            # Espera pequeña entre requests para evitar bloqueos
            time.sleep(random.uniform(1, 2))
