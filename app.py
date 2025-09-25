import requests
from io import BytesIO
from PIL import Image
import cv2, numpy as np
import pytesseract

# === Ajusta si no tienes Tesseract en PATH ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_URL = "https://appscvsgen.supercias.gob.ec/consultaCompanias/tmp/20128355289982139167780248335891.png"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://appscvsgen.supercias.gob.ec/consultaCompanias/",
    "Accept-Language": "es-EC,es;q=0.9,en;q=0.8",
}
COOKIES = {
    # "JSESSIONID": "SI REQUIERE SESION, PEGA TU COOKIE AQUI"
}

# ---------- Utilidades ----------
def keep_digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def resize3(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    return pil_img.resize((w*3, h*3), resample=Image.BICUBIC)

# ---------- Descarga ----------
def download_image_bytes(url):
    r = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=20, allow_redirects=True)
    open("debug_response.bin", "wb").write(r.content)
    print("Status:", r.status_code, "Content-Type:", r.headers.get("Content-Type"), "Len:", len(r.content))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "image" not in ctype and not r.content.startswith(b"\x89PNG\r\n\x1a\n"):
        open("debug_response.html", "wb").write(r.content)
        raise RuntimeError("No llegó imagen. Revisa debug_response.html")
    return r.content

def pil_from_bytes(img_bytes):
    try:
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError("Bytes no son imagen válida.")
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# ---------- Preprocesado (línea completa) ----------
def preprocess_for_line_ocr(pil_img):
    pil_img = resize3(pil_img)
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # OTSU: más estable para fondo simple; invertimos a texto oscuro sobre claro
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morfología MUY suave (evitar convertir 0 en 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    inv = cv2.bitwise_not(opened)  # texto oscuro / fondo claro
    cv2.imwrite("captcha_proc_line.png", inv)
    return Image.fromarray(inv)

# ---------- Preprocesado (segmentación por contornos) ----------
def preprocess_for_digit_segments(pil_img):
    pil_img = resize3(pil_img)
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # OTSU en modo texto BLANCO sobre fondo NEGRO (útil para contornos)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Nada de closing aquí (puede “partir” 0 en 8). Solo un open muy leve si hay puntitos:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite("captcha_proc_segment.png", cleaned)
    return cleaned  # texto claro (255)

# ---------- OCR (línea completa) ----------
def ocr_line(pil_img):
    cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 -c load_system_dawg=0 -c load_freq_dawg=0'
    raw = pytesseract.image_to_string(pil_img, config=cfg)
    return keep_digits(raw)

# ---------- Huecos en un recorte ----------
def count_holes(bin_crop_white_fg):
    """
    bin_crop_white_fg: binario con texto blanco (255) sobre fondo negro (0)
    Usamos jerarquía de contornos: los contornos con parent != -1 son huecos.
    """
    cnts, hier = cv2.findContours(bin_crop_white_fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    holes = 0
    for i, h in enumerate(hier[0]):
        parent = h[3]
        if parent != -1:
            holes += 1
    return holes

# ---------- OCR (por dígito con contornos + corrección 0/8) ----------
def ocr_by_segments(bin_img):
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

    # Heurística: quitar “palitos” demasiado angostos
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

        # Normaliza tamaño
        target_h = 64
        scale = target_h / crop.shape[0]
        crop_resized = cv2.resize(crop, (int(crop.shape[1]*scale), target_h), interpolation=cv2.INTER_CUBIC)

        # OCR: invertir a oscuro sobre claro
        crop_inv = cv2.bitwise_not(crop_resized)
        pil_digit = Image.fromarray(crop_inv)
        cfg = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789 -c load_system_dawg=0 -c load_freq_dawg=0'
        ch = keep_digits(pytesseract.image_to_string(pil_digit, config=cfg))

        # === Corrección por huecos (0 vs 8) ===
        holes = count_holes(crop_resized)  # usar el binario con texto blanco
        if holes == 1 and (ch == "" or ch == "8"):
            ch = "0"
        elif holes == 2:
            ch = "8"  # si Tesseract dijo 0 pero hay 2 huecos, forzamos 8

        if len(ch) == 1:
            digits.append(ch)
        elif len(ch) > 1:
            digits.append(ch[0])
        else:
            # si no reconoció, intenta una segunda pasada con psm 13
            cfg2 = r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'
            ch2 = keep_digits(pytesseract.image_to_string(pil_digit, config=cfg2))
            digits.append(ch2[:1] if ch2 else "")

    return "".join(d for d in digits if d)

# ---------- Selección final a 6 dígitos ----------
def choose_best_six(line_guess: str, seg_guess: str) -> str:
    candidates = []
    for g in [line_guess, seg_guess]:
        if not g:
            continue
        g = keep_digits(g)
        if len(g) > 6:
            # recorte conservador: quita un '1' suelto si hay >1
            if g.count('1') >= 2:
                g = g.replace('1', '', 1)
        g = g[:6]
        candidates.append(g)

    exact_six = [c for c in candidates if len(c) == 6]
    if exact_six:
        # Heurística: penaliza '8' si el otro candidato tiene '0' en esa posición
        # (ya que 0→8 es el error típico)
        def penalty(s):
            return s.count('8')
        exact_six.sort(key=penalty)
        return exact_six[0]

    return (max(candidates, key=len)[:6] if candidates else "")

# ---------- Main ----------
def main():
    try:
        img_bytes = download_image_bytes(IMAGE_URL)
    except Exception as e:
        print("Error descargando la imagen:", e)
        print("Revisa: debug_response.bin y/o debug_response.html")
        return

    pil_raw = pil_from_bytes(img_bytes)
    pil_raw.save("captcha_raw.png")

    # Pasada A (línea)
    pil_line = preprocess_for_line_ocr(pil_raw)
    guess_line = ocr_line(pil_line)
    print("LINE (raw):", guess_line)

    # Pasada B (segmentos con corrección morfológica 0/8)
    bin_seg = preprocess_for_digit_segments(pil_raw)
    guess_seg = ocr_by_segments(bin_seg)
    print("SEGS (raw):", guess_seg)

    final6 = choose_best_six(guess_line, guess_seg)
    print("RESULTADO (6 dígitos):", final6)

    print("Archivos:", "captcha_raw.png", "captcha_proc_line.png", "captcha_proc_segment.png", "debug_response.bin")

if __name__ == "__main__":
    main()
