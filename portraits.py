import argparse
import requests
import numpy as np
import cv2
import io
import os
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

# ======================
# CONFIG
# ======================

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/original"

WIDTH = 1600
HEIGHT = 900
FONT_PATH = "fonts/Roboto-Bold.ttf"

# ======================
# TMDB FUNCTIONS
# ======================

def search_person(identifier):
    if str(identifier).isdigit():
        res = requests.get(
            f"{BASE_URL}/person/{identifier}",
            params={"api_key": TMDB_API_KEY}
        ).json()
        return res if "id" in res else None

    res = requests.get(
        f"{BASE_URL}/search/person",
        params={"api_key": TMDB_API_KEY, "query": identifier}
    ).json()

    for p in res.get("results", []):
        if p["name"].lower() == str(identifier).lower():
            return p

    return res["results"][0] if res.get("results") else None

def get_best_profile(person_id):
    res = requests.get(
        f"{BASE_URL}/person/{person_id}/images",
        params={"api_key": TMDB_API_KEY}
    ).json()

    profiles = res.get("profiles", [])
    if not profiles:
        return None

    best = sorted(profiles, key=lambda x: x["height"], reverse=True)[0]
    return best["file_path"]

def download_image(path):
    url = IMAGE_BASE + path
    return requests.get(url).content

# ======================
# IMAGE PROCESSING
# ======================

def feather_edges(subject):
    alpha = np.array(subject.split()[-1]).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
    subject.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return subject

def fade_right_edge(img):
    # Fades the right side of the subject into transparency so there isn't a hard cut
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    
    # Create a gradient mask that goes from 1 to 0 on the right side
    fade_start = int(w * 0.70)
    for x in range(fade_start, w):
        fade_factor = 1.0 - ((x - fade_start) / (w - fade_start))
        alpha[:, x] *= fade_factor
        
    arr[:, :, 3] = alpha
    return Image.fromarray(arr.astype(np.uint8))

def enhance_contrast(img):
    img_np = np.array(img)
    img_np = cv2.convertScaleAbs(img_np, alpha=1.2, beta=10)
    return Image.fromarray(img_np)

def add_face_light(image):
    img = np.array(image).astype(np.float32)
    h, w = img.shape[:2]

    cx, cy = int(w * 0.35), int(h * 0.45)
    mask = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx = (x - cx) / (w * 0.5)
            dy = (y - cy) / (h * 0.5)
            dist = np.sqrt(dx*dx + dy*dy)
            mask[y, x] = np.exp(-dist * 2.5)

    mask = cv2.GaussianBlur(mask, (201, 201), 0)

    for i in range(3):
        img[:, :, i] += mask * 60

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

def create_gradient_background():
    bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    bg[:, :] = (6, 7, 7)
    return Image.fromarray(bg)

def add_film_grain(image):
    img = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 6, img.shape)
    img += noise
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

def draw_text_left(draw, text, x, y, font):
    lines = text.split("\n")
    y_offset = 0
    for line in lines:
        draw.text((x, y + y_offset), line, font=font, fill=(180, 180, 180)) 
        y_offset += int(font.size * 0.90)

# ======================
# MAIN POSTER FUNCTION
# ======================

def create_poster(image_bytes, name, output):
    subject = Image.open(io.BytesIO(remove(image_bytes))).convert("RGBA")
    subject = feather_edges(subject)
    subject = fade_right_edge(subject)
    
    subject.thumbnail((1100, 1200))

    alpha = subject.getchannel("A")
    gray = subject.convert("L")
    subject = Image.merge("RGB", (gray, gray, gray))

    subject = enhance_contrast(subject)
    subject = add_face_light(subject)
    subject.putalpha(alpha)

    bg = create_gradient_background()

    x_offset = -50
    y_offset = max(0, HEIGHT - subject.height + 50)
    bg.paste(subject, (x_offset, y_offset), subject)

    draw = ImageDraw.Draw(bg)
    try:
        font = ImageFont.truetype(FONT_PATH, 140)
    except OSError:
        import urllib.request
        os.makedirs("fonts", exist_ok=True)
        urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf", "fonts/Roboto-Bold.ttf")
        font = ImageFont.truetype("fonts/Roboto-Bold.ttf", 140)

    draw_text_left(draw, name.title().replace(" ", "\n"), int(WIDTH * 0.45), int(HEIGHT * 0.40), font)

    bg = add_film_grain(bg)
    bg.save(output, quality=95)

# ======================
# CLI
# ======================

def process_person(identifier):
    print(f"Processing: {identifier}")

    person = search_person(identifier)
    if not person:
        print("Not found")
        return

    img_path = get_best_profile(person["id"])
    if not img_path:
        print("No image")
        return

    image_bytes = download_image(img_path)

    name = person["name"]
    filename = name.replace(" ", "_") + ".jpg"

    create_poster(image_bytes, name, filename)
    print(f"Saved: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Cinematic Portrait Generator")
    parser.add_argument("--person", action="append", required=True)
    parser.add_argument("--size", required=False)

    args = parser.parse_args()

    for identifier in args.person:
        process_person(identifier)

if __name__ == "__main__":
    main()
