# Imagen base ligera con Python
FROM python:3.11-slim

# Variables de entorno básicas
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Instalar Tesseract y librerías nativas necesarias para OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias (primero requirements para aprovechar cache de Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Render asigna el puerto en la variable de entorno PORT
EXPOSE 10000

# Comando de arranque: gunicorn con uvicorn worker
# Cambia "app:app" si tu archivo se llama distinto (ej: index:app)
CMD sh -c 'gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:${PORT:-10000} app:app'
