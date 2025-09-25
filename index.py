import os
from typing import Optional, Dict

import requests
from io import BytesIO
from PIL import Image
import cv2, numpy as np
import pytesseract

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# =========================
# Configuración Tesseract
# =========================
# Si Tesseract no está en PATH, usa variable de entorno TESSERACT_CMD o ajusta ruta fija:
TESS_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.isfile(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD
# Si no existe el archivo, pytesseract intentará usar el PATH del sistema.

# =========================
# FastAPI
# =========================
app = FastAPI(title="OCR 6 dígitos", version="1.0.0")

class OCRRequest(BaseModel):
    url: HttpUrl
    cookies: Optional[Dict[str, str]] = None
    referer: Optional[str] = "https://appscvsgen.supercias.gob.ec/consultaCompanias/"

class OCRResponse(BaseModel):
    ok: bool
    digits: Optional[str] = None
    length: Optional[int] = None
    line_guess: Optional[str] = None
    seg_guess: Optional[str] = None
    error: Optional[str] = None

# =========================
# Utilidades OCR
# =========================
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "es-EC,es;q=0.9,en;q=0.8",
}

def keep_digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def resize3(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    return pil_img.resize((w*3, h*3), resample=Image.BICUBIC)

def download_image_bytes(url: str, referer: Optional[str], cookies: Optional[Dict[str, str]]) -> bytes:
    headers = DEFAULT_HEADERS.copy()
    if referer:
        headers["Referer"] = referer
    r = requests.get(url, headers=headers, cookies=cookies or {}, timeout=20, allow_redirects=True)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HTTP {r.status_code} al pedir la imagen")
    ctype = (r.headers.get("Content-Type") or "").lower()
    content = r.content
    # Acepta contenido image/* o PNG por firma binaria
    if "image" not in ctype and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=400, detail="La URL no devolvió una imagen (¿sesión/cookies requeridas?).")
    return content

def pil_from_bytes(img_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Bytes no representan una imagen válida.")
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# ---- Preprocesado (línea completa, conservador para no convertir 0->8) ----
def preprocess_for_line_ocr(pil_img: Image.Image) -> Image.Image:
    pil_img = resize3(pil_img)
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    inv = cv2.bitwise_not(opened)  # texto oscuro sobre fondo claro
    return Image.fromarray(inv)

# ---- Preprocesado (segmentación por contornos) ----
def preprocess_for_digit_segments(pil_img: Image.Image) -> np.ndarray:
    pil_img = resize3(pil_img)
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # texto blanco sobre fondo negro
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def ocr_line(pil_img: Image.Image) -> str:
    cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 -c load_system_dawg=0 -c load_freq_dawg=0'
    raw = pytesseract.image_to_string(pil_img, config=cfg)
    return keep_digits(raw)

def count_holes(bin_crop_white_fg: np.ndarray) -> int:
    cnts, hier = cv2.findContours(bin_crop_white_fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    holes = 0
    for h in hier[0]:
        parent = h[3]
        if parent != -1:
            holes += 1
    return holes

def ocr_by_segments(bin_img: np.ndarray) -> str:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = bin_img.shape[:2]
    MIN_H = H * 0.25
    MIN_W = W * 0.015
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h >= MIN_H and w >= MIN_W:
            boxes.append((x, y, w, h))
    if not boxes:
        return ""
    boxes.sort(key=lambda b: b[0])
    widths = [w for (_,_,w,_) in boxes]
    median_w = np.median(widths) if widths else 0
    NARROW_DROP_RATIO = 0.45
    filtered = [(x,y,w,h) for (x,y,w,h) in boxes if w >= median_w * NARROW_DROP_RATIO] or boxes

    digits = []
    for (x,y,w,h) in filtered:
        pad = int(0.20 * max(w,h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = bin_img[y0:y1, x0:x1]  # texto BLANCO

        target_h = 64
        scale = target_h / crop.shape[0]
        crop_resized = cv2.resize(crop, (int(crop.shape[1]*scale), target_h), interpolation=cv2.INTER_CUBIC)

        # OCR: invertir a oscuro sobre claro
        crop_inv = cv2.bitwise_not(crop_resized)
        pil_digit = Image.fromarray(crop_inv)
        cfg = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789 -c load_system_dawg=0 -c load_freq_dawg=0'
        ch = keep_digits(pytesseract.image_to_string(pil_digit, config=cfg))

        # Corrección morfológica 0 vs 8
        holes = count_holes(crop_resized)  # usar binario blanco
        if holes == 1 and (ch == "" or ch == "8"):
            ch = "0"
        elif holes == 2:
            ch = "8"

        if len(ch) == 1:
            digits.append(ch)
        elif len(ch) > 1:
            digits.append(ch[0])
        else:
            cfg2 = r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'
            ch2 = keep_digits(pytesseract.image_to_string(pil_digit, config=cfg2))
            digits.append(ch2[:1] if ch2 else "")

    return "".join(d for d in digits if d)

def choose_best_six(line_guess: str, seg_guess: str) -> str:
    candidates = []
    for g in (line_guess, seg_guess):
        if not g:
            continue
        g = keep_digits(g)
        if len(g) > 6 and g.count('1') >= 2:
            g = g.replace('1', '', 1)
        g = g[:6]
        candidates.append(g)

    exact = [c for c in candidates if len(c) == 6]
    if exact:
        exact.sort(key=lambda s: s.count('8'))  # penaliza 8's por el típico 0->8
        return exact[0]
    return (max(candidates, key=len)[:6] if candidates else "")

def process_url(url: str, cookies: Optional[Dict[str, str]], referer: Optional[str]) -> Dict[str, str]:
    img_bytes = download_image_bytes(url, referer, cookies)
    pil_raw = pil_from_bytes(img_bytes)

    pil_line = preprocess_for_line_ocr(pil_raw)
    guess_line = ocr_line(pil_line)

    bin_seg = preprocess_for_digit_segments(pil_raw)
    guess_seg = ocr_by_segments(bin_seg)

    final6 = choose_best_six(guess_line, guess_seg)
    if not final6 or len(final6) != 6:
        raise HTTPException(status_code=422, detail=f"No se pudo obtener 6 dígitos (line='{guess_line}', seg='{guess_seg}').")

    return {"digits": final6, "line_guess": guess_line, "seg_guess": guess_seg}

# =========================
# Rutas
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ocr-url", response_model=OCRResponse)
def ocr_url(payload: OCRRequest):
    try:
        result = process_url(payload.url, payload.cookies, payload.referer)
        return OCRResponse(ok=True, digits=result["digits"], length=6,
                           line_guess=result["line_guess"], seg_guess=result["seg_guess"])
    except HTTPException as e:
        return OCRResponse(ok=False, error=e.detail)
    except pytesseract.TesseractNotFoundError:
        return OCRResponse(ok=False, error="Tesseract no está instalado o no está en PATH/TESSERACT_CMD.")
    except Exception as e:
        return OCRResponse(ok=False, error=str(e))
