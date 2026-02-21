# app.py
import os
import re
from io import BytesIO

import requests
import numpy as np
import cv2
from PIL import Image
import pytesseract

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl


# =========================================================
# Tesseract path
# =========================================================
TESS_ENV = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.isfile(TESS_ENV):
    pytesseract.pytesseract.tesseract_cmd = TESS_ENV
elif os.path.isfile("/usr/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
elif os.path.isfile(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="OCR URL Simple", version="1.4.0")

DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))

# Permite MAYÚSCULAS + minúsculas + números
WHITELIST = os.getenv(
    "OCR_WHITELIST",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)

DEFAULT_HEADERS = {
    "User-Agent": os.getenv(
        "OCR_UA",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "es-EC,es;q=0.9,en;q=0.8",
}


# =========================================================
# Models
# =========================================================
class OCRUrlRequest(BaseModel):
    url: HttpUrl

class OCRResponse(BaseModel):
    ok: bool
    text: str | None = None
    error: str | None = None
    line_guess: str | None = None
    seg_guess: str | None = None


# =========================================================
# Helpers
# =========================================================
def keep_alnum_keep_case(s: str) -> str:
    s = (s or "")
    return re.sub(r"[^A-Za-z0-9]", "", s)

def resize3(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    return pil_img.resize((max(1, w * 3), max(1, h * 3)), resample=Image.BICUBIC)

def pil_from_bytes(img_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Los bytes no representan una imagen válida.")
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def download_image_bytes(url: str) -> bytes:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=TIMEOUT, allow_redirects=True)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HTTP {r.status_code} al descargar la imagen")

    ctype = (r.headers.get("Content-Type") or "").lower()
    content = r.content

    if "image" not in ctype and not (
        content.startswith(b"\x89PNG\r\n\x1a\n") or content[:2] == b"\xff\xd8"
    ):
        if DEBUG:
            try:
                with open("debug_not_image.bin", "wb") as f:
                    f.write(content)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail="La URL no devolvió una imagen (debe ser pública y directa).")

    return content


# =========================================================
# Preprocesados (varios)
# =========================================================
def to_gray_resized(pil_img: Image.Image) -> np.ndarray:
    pil_img = resize3(pil_img)
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def prep_otsu(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    if np.mean(g) < 127:
        g = cv2.bitwise_not(g)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return th

def prep_adaptive(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    if np.mean(g) < 127:
        g = cv2.bitwise_not(g)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 4)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return th

def prep_sharpen_otsu(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, g, -0.6, 0)
    if np.mean(sharp) < 127:
        sharp = cv2.bitwise_not(sharp)
    _, th = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # un cierre suave ayuda a “cerrar” loops de letras
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return th

def prep_for_segments(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return th


# =========================================================
# OCR con confianza (línea)
# =========================================================
def ocr_line_with_confidence(bin_img: np.ndarray) -> tuple[str, float]:
    pil_img = Image.fromarray(bin_img)
    cfg = (
        f'--oem 3 --psm 7 '
        f'-c tessedit_char_whitelist={WHITELIST} '
        f'-c load_system_dawg=0 -c load_freq_dawg=0'
    )

    data = pytesseract.image_to_data(pil_img, config=cfg, output_type=pytesseract.Output.DICT)
    texts = []
    confs = []
    for t, c in zip(data.get("text", []), data.get("conf", [])):
        try:
            cf = float(c)
        except Exception:
            continue
        t2 = keep_alnum_keep_case(t)
        if t2:
            texts.append(t2)
            if cf >= 0:
                confs.append(cf)

    out = "".join(texts)
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    return out, avg_conf

def best_line_guess(pil_raw: Image.Image) -> tuple[str, float]:
    gray = to_gray_resized(pil_raw)

    variants = [
        ("otsu", prep_otsu(gray)),
        ("adaptive", prep_adaptive(gray)),
        ("sharp_otsu", prep_sharpen_otsu(gray)),
    ]

    best_text = ""
    best_score = -1.0

    for name, bin_img in variants:
        txt, conf = ocr_line_with_confidence(bin_img)
        # score: prioriza longitud y confianza
        score = (len(txt) * 10.0) + conf
        if DEBUG:
            cv2.imwrite(f"debug_{name}.png", bin_img)
        if score > best_score:
            best_score = score
            best_text = txt

    return best_text, best_score


# =========================================================
# OCR por segmentación (fallback)
# =========================================================
def ocr_by_segments(bin_inv: np.ndarray) -> str:
    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bin_inv.shape[:2]
    boxes = []

    MIN_H = H * 0.25
    MIN_W = W * 0.008
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h >= MIN_H and w >= MIN_W:
            boxes.append((x, y, w, h))

    if not boxes:
        return ""

    boxes.sort(key=lambda b: b[0])

    out = []
    for (x, y, w, h) in boxes:
        pad = int(0.20 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

        crop = bin_inv[y0:y1, x0:x1]  # blanco sobre negro
        target_h = 64
        scale = target_h / max(1, crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), target_h), interpolation=cv2.INTER_CUBIC)

        crop_for_ocr = cv2.bitwise_not(crop)
        pil_char = Image.fromarray(crop_for_ocr)

        cfg = (
            f'--oem 3 --psm 10 '
            f'-c tessedit_char_whitelist={WHITELIST} '
            f'-c load_system_dawg=0 -c load_freq_dawg=0'
        )
        raw = pytesseract.image_to_string(pil_char, config=cfg)
        ch = keep_alnum_keep_case(raw)
        if ch:
            out.append(ch[0])

    return "".join(out)


# =========================================================
# Orquestador
# =========================================================
def ocr_from_url(url: str) -> dict:
    img_bytes = download_image_bytes(url)
    pil_raw = pil_from_bytes(img_bytes)

    # 1) Mejor resultado por línea (multi-pass + confianza)
    guess_line, _score = best_line_guess(pil_raw)

    # 2) Segmentación como fallback (a veces falla, pero ayuda)
    gray = to_gray_resized(pil_raw)
    bin_seg = prep_for_segments(gray)
    if DEBUG:
        cv2.imwrite("debug_segments.png", bin_seg)
    guess_seg = ocr_by_segments(bin_seg)

    # Elige el más largo (y normalmente el de línea ya viene mejor)
    final_text = guess_line if len(guess_line) >= len(guess_seg) else guess_seg

    if not final_text:
        raise HTTPException(status_code=422, detail=f"No se pudo reconocer texto (line='{guess_line}', seg='{guess_seg}').")

    return {"text": final_text, "line_guess": guess_line, "seg_guess": guess_seg}


# =========================================================
# Rutas
# =========================================================
@app.get("/")
def root():
    return {"name": "ocr-url-simple", "version": "1.4.0", "endpoints": ["/health", "/ocr-url"]}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ocr-url", response_model=OCRResponse)
def ocr_url(payload: OCRUrlRequest):
    try:
        r = ocr_from_url(str(payload.url))
        return OCRResponse(ok=True, text=r["text"], line_guess=r["line_guess"], seg_guess=r["seg_guess"])
    except HTTPException as e:
        return OCRResponse(ok=False, error=str(e.detail))
    except pytesseract.TesseractNotFoundError:
        return OCRResponse(ok=False, error="Tesseract no está instalado o no está en PATH/TESSERACT_CMD.")
    except Exception as e:
        return OCRResponse(ok=False, error=str(e))


# =========================================================
# Run direct: python app.py
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
