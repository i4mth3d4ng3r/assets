import argparse
import requests
import numpy as np
import cv2
import io
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from rembg import remove

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/original"

WIDTH = 1600
HEIGHT = 900
FONT_PATH = "fonts/Oswald-Bold.ttf"
SUBJECT_HEIGHT_RATIO = 1.03

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

def feather_edges(subject):
    alpha = np.array(subject.getchannel("A")).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
    subject.putalpha(Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)))
    return subject

def scale_to_height(img, target_height):
    orig_w, orig_h = img.size
    scale = target_height / orig_h
    new_w = int(orig_w * scale)
    return img.resize((new_w, target_height), Image.LANCZOS)

def enhance_contrast(img):
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 255.0) ** 0.92 * 255
    img_np = cv2.convertScaleAbs(img_np, alpha=1.18, beta=-10)
    return Image.fromarray(img_np)

def add_face_light(image):
    img = np.array(image).astype(np.float32)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    nx = xx / w
    ny = yy / h

    light = 1.0 - (nx * 0.9 + ny * 0.45)
    light = np.clip(light, 0, 1)
    light = cv2.GaussianBlur(light, (0, 0), 55)

    for i in range(3):
        img[:, :, i] *= (0.90 + light * 0.22)

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

def fade_right_edge_canvas(subject, x_offset):
    arr = np.array(subject).astype(np.float32)
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]

    yy, xx = np.mgrid[0:h, 0:w]
    canvas_x = (xx + x_offset) / WIDTH
    canvas_y = yy / h

    start = 0.29
    end = 0.53
    t = np.clip((canvas_x - start) / (end - start), 0, 1)
    t = t * t * (3 - 2 * t)

    center_bias = 1.0 - np.abs(canvas_y - 0.52) * 0.22
    fade = 1.0 - (t * center_bias)

    alpha *= np.clip(fade, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 9)

    arr[:, :, 3] = np.clip(alpha, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def create_gradient_background():
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    nx = xx / WIDTH
    ny = yy / HEIGHT

    # Mild left-to-right falloff, but keep the right side lifted
    base = 15.5 + (1.0 - nx) * 6.5

    left_glow = np.exp(-(((nx - 0.15) / 0.19) ** 2 + ((ny - 0.46) / 0.36) ** 2)) * 10.0
    upper_left_lift = np.exp(-(((nx - 0.11) / 0.18) ** 2 + ((ny - 0.18) / 0.20) ** 2)) * 3.5
    text_area_lift = np.exp(-(((nx - 0.66) / 0.24) ** 2 + ((ny - 0.46) / 0.34) ** 2)) * 2.8

    bands = (
        np.sin(nx * WIDTH * 0.055 + 0.8) * 0.35 +
        np.sin(nx * WIDTH * 0.017 + 1.7) * 0.22
    )

    bg = base + left_glow + upper_left_lift + text_area_lift + bands
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 2.4)

    rgb = np.dstack([bg, bg, bg])
    rgb = np.clip(rgb, 10, 30).astype(np.uint8)

    return Image.fromarray(rgb)

def add_film_grain(image):
    img = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 1.25, img.shape)
    img += noise
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

def draw_text_left(draw, text, x, y, font, line_gap=0.855, fill=(222, 222, 222)):
    lines = text.split("\n")
    y_offset = 0
    for line in lines:
        draw.text((x, y + y_offset), line, font=font, fill=fill)
        y_offset += int(font.size * line_gap)

def create_poster(image_bytes, name, output):
    subject = Image.open(io.BytesIO(remove(image_bytes))).convert("RGBA")
    subject = feather_edges(subject)

    target_h = int(HEIGHT * SUBJECT_HEIGHT_RATIO)
    subject = scale_to_height(subject, target_h)

    alpha = subject.getchannel("A")
    gray = subject.convert("L")
    subject_rgb = Image.merge("RGB", (gray, gray, gray))

    subject_rgb = enhance_contrast(subject_rgb)
    subject_rgb = add_face_light(subject_rgb)
    subject_rgb.putalpha(alpha)
    subject = subject_rgb

    x_offset = int(WIDTH * 0.06)
    y_offset = HEIGHT - subject.height

    subject = fade_right_edge_canvas(subject, x_offset)

    bg = create_gradient_background()
    bg.paste(subject, (x_offset, y_offset), subject)

    draw = ImageDraw.Draw(bg)

    try:
        font = ImageFont.truetype(FONT_PATH, 114)
    except OSError:
        try:
            font = ImageFont.truetype("fonts/Oswald.ttf", 114)
        except OSError:
            font = ImageFont.load_default()

    draw_text_left(
        draw,
        name.title().replace(" ", "\n"),
        int(WIDTH * 0.545),
        int(HEIGHT * 0.255),
        font,
        line_gap=0.855,
        fill=(222, 222, 222)
    )

    bg = add_film_grain(bg)
    bg.save(output, quality=95)

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