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
FONT_PATH = "arial.ttf"

# ======================
# TMDB FUNCTIONS
# ======================

def search_person(identifier):
    # Check if the input is an ID number
    if str(identifier).isdigit():
        res = requests.get(
            f"{BASE_URL}/person/{identifier}",
            params={"api_key": TMDB_API_KEY}
        ).json()
        return res if "id" in res else None

    # Otherwise, search by name
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


def enhance_contrast(img):
    img_np = np.array(img)
    img_np = cv2.convertScaleAbs(img_np, alpha=1.4, beta=-20)
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
            mask[y, x] = np.exp(-dist * 3)

    mask = cv2.GaussianBlur(mask, (201, 201), 0)

    for i in range(3):
        img[:, :, i] += mask * 60

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def create_gradient_background():
    bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for x in range(WIDTH):
        val = int(20 + (x / WIDTH) * 40)
        bg[:, x] = (val, val, val)

    return Image.fromarray(bg)


def add_vignette(image):
    w, h = image.size
    vignette = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx = (x - w / 2) / (w / 2)
            dy = (y - h / 2) / (h / 2)
            vignette[y, x] = 1 - min(np.sqrt(dx*dx + dy*dy), 1)

    vignette = cv2.GaussianBlur(vignette, (201, 201), 0)

    img = np.array(image).astype(np.float32)
    for i in range(3):
        img[:, :, i] *= vignette

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def add_text_glow(bg):
    overlay = np.zeros_like(np.array(bg)).astype(np.float32)
    h, w = overlay.shape[:2]

    for x in range(w):
        strength = max(0, (x - w * 0.55) / (w * 0.45))
        overlay[:, x] += strength * 25

    overlay = cv2.GaussianBlur(overlay, (301, 301), 0)

    result = np.array(bg).astype(np.float32) + overlay
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def add_film_grain(image):
    img = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 8, img.shape)
    img += noise
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def draw_text(draw, text, x, y, font):
    lines = text.split("\n")
    y_offset = 0

    for line in lines:
        draw.text((x, y + y_offset), line, font=font, fill=(235,235,235))
        y_offset += int(font.size * 0.85)


# ======================
# MAIN POSTER FUNCTION
# ======================

def create_poster(image_bytes, name, output):
    # Remove BG
    subject = Image.open(io.BytesIO(remove(image_bytes))).convert("RGBA")

    subject = feather_edges(subject)
    subject.thumbnail((700, 800))

    # Save transparency mask before applying grayscale and lighting effects
    alpha = subject.getchannel("A")

    # grayscale
    gray = subject.convert("L")
    subject = Image.merge("RGB", (gray, gray, gray))

    subject = enhance_contrast(subject)
    subject = add_face_light(subject)

    # Re-apply the transparency mask before pasting
    subject.putalpha(alpha)

    bg = create_gradient_background()

    # paste subject
    x_offset = 50
    y_offset = HEIGHT - subject.height
    bg.paste(subject, (x_offset, y_offset), subject)

    bg = add_text_glow(bg)
    bg = add_vignette(bg)

    draw = ImageDraw.Draw(bg)
    font = ImageFont.truetype(FONT_PATH, 120)

    draw_text(draw, name.replace(" ", "\n"), WIDTH//2 + 100, HEIGHT//2 - 80, font)

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

    # Use the actual name for the text and filename, not the ID
    name = person["name"]
    filename = name.replace(" ", "_") + ".jpg"

    create_poster(image_bytes, name, filename)

    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Cinematic Portrait Generator")
    parser.add_argument("--person", action="append", required=True)
    # Added so the GitHub Action doesn't crash if `--size 3840x2160` is passed
    parser.add_argument("--size", required=False)

    args = parser.parse_args()

    for identifier in args.person:
        process_person(identifier)


if __name__ == "__main__":
    main()