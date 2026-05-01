import argparse
import requests
import numpy as np
import cv2
import io
import os
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import mediapipe as mp

# ======================
# CONFIG
# ======================

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/original"

DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 900
FONT_PATH = "arial.ttf"

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ======================
# HELPERS
# ======================

def parse_size(size_str):
    return map(int, size_str.lower().split("x"))

def get_api_key(cli_key):
    return cli_key or os.getenv("TMDB_API_KEY") or exit("API key required")

def is_id(value):
    return str(value).isdigit()

# ======================
# TMDB
# ======================

def get_person(identifier, api_key):
    if is_id(identifier):
        res = requests.get(f"{BASE_URL}/person/{identifier}", params={"api_key": api_key}).json()
        return res if "id" in res else None

    res = requests.get(f"{BASE_URL}/search/person",
        params={"api_key": api_key, "query": identifier}).json()

    return res["results"][0] if res.get("results") else None


def get_best_profile(person_id, api_key):
    res = requests.get(f"{BASE_URL}/person/{person_id}/images",
        params={"api_key": api_key}).json()

    profiles = res.get("profiles", [])
    if not profiles:
        return None

    return sorted(profiles, key=lambda x: x["height"], reverse=True)[0]["file_path"]


def download_image(path):
    return requests.get(IMAGE_BASE + path).content

# ======================
# FACE + EYE DETECTION
# ======================

def detect_face_and_eyes(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    results = mp_face.process(img)
    if not results.detections:
        return None

    det = results.detections[0]
    bbox = det.location_data.relative_bounding_box

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    kp = det.location_data.relative_keypoints
    eye_x = int(((kp[0].x + kp[1].x)/2) * w)
    eye_y = int(((kp[0].y + kp[1].y)/2) * h)

    return {"face": (x,y,bw,bh), "eye_center": (eye_x, eye_y)}


def smart_crop(pil_img, target_ratio=16/9):
    data = detect_face_and_eyes(pil_img)
    if not 
        return pil_img

    img_w, img_h = pil_img.size
    eye_x, eye_y = data["eye_center"]

    target_eye_y = int(img_h * 0.38)
    dy = eye_y - target_eye_y

    crop_h = int(img_h * 0.85)
    crop_w = int(crop_h * target_ratio)

    cx = int(eye_x - crop_w * 0.25)
    cy = int((img_h // 2) + dy)

    left = max(0, cx - crop_w // 2)
    top = max(0, cy - crop_h // 2)
    right = min(img_w, left + crop_w)
    bottom = min(img_h, top + crop_h)

    return pil_img.crop((left, top, right, bottom))

# ======================
# IMAGE FX
# ======================

def feather_edges(subject):
    alpha = np.array(subject.split()[-1]).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (21,21), 0)
    subject.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return subject


def enhance_contrast(img):
    return Image.fromarray(cv2.convertScaleAbs(np.array(img), alpha=1.3, beta=0))


# 🔥 NEW: AI-style relighting
def relight_face(image):
    img = np.array(image).astype(np.float32)
    h, w = img.shape[:2]

    # simulate key light from top-left
    light = np.zeros((h, w), np.float32)

    for y in range(h):
        for x in range(w):
            dx = x / w
            dy = y / h
            val = 1.2 - (dx * 0.8 + dy * 0.6)
            light[y, x] = max(0, val)

    light = cv2.GaussianBlur(light, (201,201), 0)

    img += np.stack([light * 75]*3, axis=-1)

    # shadow recovery
    img = np.clip(img * 1.1 + 20, 0, 255)

    return Image.fromarray(img.astype(np.uint8))


def add_vignette(image):
    w,h = image.size
    vignette = np.zeros((h,w), np.float32)

    for y in range(h):
        for x in range(w):
            dx = (x-w/2)/(w/2)
            dy = (y-h/2)/(h/2)
            vignette[y,x] = 1-min(np.sqrt(dx*dx+dy*dy),1)

    vignette = cv2.GaussianBlur(vignette,(201,201),0)

    img = np.array(image).astype(np.float32)
    img *= vignette[...,None]

    return Image.fromarray(np.clip(img,0,255).astype(np.uint8))


def add_text_glow(bg):
    overlay = np.zeros_like(np.array(bg)).astype(np.float32)
    h,w = overlay.shape[:2]

    for x in range(w):
        strength = max(0,(x-w*0.55)/(w*0.45))
        overlay[:,x]+=strength*25

    overlay = cv2.GaussianBlur(overlay,(301,301),0)

    return Image.fromarray(np.clip(np.array(bg)+overlay,0,255).astype(np.uint8))


def add_film_grain(img):
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0,8,arr.shape)
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))


# 🔥 NEW: auto font scaling
def fit_font(draw, text, max_width, base_size):
    size = base_size
    while size > 20:
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = draw.multiline_textbbox((0,0), text, font=font, spacing=int(size*0.85))
        if bbox[2] <= max_width:
            return font
        size -= 4
    return ImageFont.truetype(FONT_PATH, size)


def draw_text(draw, text, x, y, font):
    y_offset = 0
    for line in text.split("\n"):
        draw.text((x, y+y_offset), line, font=font, fill=(235,235,235))
        y_offset += int(font.size * 0.85)

# ======================
# POSTER
# ======================

def create_poster(image_bytes, name, output, width, height):
    scale = width / DEFAULT_WIDTH

    subject = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    subject = smart_crop(subject)

    subject = Image.open(io.BytesIO(remove(subject.tobytes()))).convert("RGBA")

    subject = feather_edges(subject)
    subject.thumbnail((int(900*scale), int(1000*scale)))

    gray = subject.convert("L")
    subject = Image.merge("RGB",(gray,gray,gray))

    subject = enhance_contrast(subject)

    # 🔥 NEW RELIGHT
    subject = relight_face(subject)

    # background
    bg = np.zeros((height,width,3),dtype=np.uint8)
    for x in range(width):
        val = int(20+(x/width)*40)
        bg[:,x]=(val,val,val)

    bg = Image.fromarray(bg)

    x_offset = int(50*scale)
    y_offset = height - subject.height
    bg.paste(subject,(x_offset,y_offset),subject)

    bg = add_text_glow(bg)
    bg = add_vignette(bg)

    draw = ImageDraw.Draw(bg)

    text = name.replace(" ","\n")

    font = fit_font(draw, text, int(width*0.35), int(120*scale))

    draw_text(draw, text, int(width*0.62), int(height*0.45), font)

    bg = add_film_grain(bg)
    bg.save(output, quality=95)

# ======================
# PROCESS
# ======================

def process_person(identifier, api_key, width, height):
    person = get_person(identifier, api_key)
    if not person:
        print(f"Not found: {identifier}")
        return

    name = person["name"]
    print(f"Processing: {name}")

    path = get_best_profile(person["id"], api_key)
    if not path:
        print("No image")
        return

    img_bytes = download_image(path)

    filename = f"{name.replace(' ','_')}_{width}x{height}.jpg"

    create_poster(img_bytes, name, filename, width, height)

    print(f"Saved: {filename}")

# ======================
# CLI
# ======================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--person", action="append", required=True,
                        help="Name or TMDB person ID")
    parser.add_argument("--api-key")
    parser.add_argument("--size", default="1600x900")

    args = parser.parse_args()

    width, height = parse_size(args.size)
    api_key = get_api_key(args.api_key)

    for p in args.person:
        process_person(p, api_key, width, height)


if __name__ == "__main__":
    main()