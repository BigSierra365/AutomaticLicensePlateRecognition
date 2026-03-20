# 🚘 LicensePlateDetector

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-yellow)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Descripción / Propósito

**LicensePlateDetector** es un sistema avanzado de Reconocimiento Automático de Matrículas (ALPR - *Automatic License Plate Recognition*) diseñado para procesar lotes de imágenes, detectar vehículos y extraer el texto de las placas con alta precisión. 

Este proyecto resuelve el problema de la transcripción manual de matrículas en sistemas de peajes, parkings o seguridad vial, automatizando la lectura y generando reportes estructurados listos para ser analizados.

## ⚙️ Stack Tecnológico

| Capa | Tecnologías | Propósito |
|---|---|---|
| **Visión Computacional** | OpenCV, imutils | Procesamiento de imágenes, redimensión y visualización (UI). |
| **Modelos de IA** | Ultralytics (YOLO) | Detección de objetos (inferencia sobre la ubicación de la matrícula). |
| **OCR** | PaddleOCR | Reconocimiento óptico de caracteres multilingüe. |
| **Data & Regex** | Python (re) | Limpieza y validación del texto extraído. |

## ✨ Características Principales

* **Detección Robusta:** Emplea redes neuronales convolucionales (YOLO) para ubicar matrículas en entornos de iluminación variable.
* **Extracción Precisa:** Utiliza *PaddleOCR* para entender la orientación del texto y transcribirlo eficazmente.
* **Procesamiento en Lotes:** Capacidad de analizar carpetas enteras de imágenes automáticamente.
* **Filtros de Validación:** Validación de lectura mediante expresiones regulares para aislar caracteres alfanuméricos válidos.
* **Generación de Reportes:** Creación de informes detallados y estructurados con marca de tiempo, exportados a archivos de texto.
* **Feedback Visual:** Interfaz en tiempo real que dibuja las *bounding boxes* y la predicción del OCR sobre la imagen original.

## 🧠 Arquitectura y Lógica

El pipeline del sistema sigue un enfoque modular en 4 etapas:
1. **Carga y Optimización:** Inicialización de pesos de los modelos YOLO y PaddleOCR en memoria para evitar latencias durante la inferencia.
2. **Preprocesamiento:** Escaneo del directorio de entrada (`/inputs`) para localizar formatos compatibles (`.jpg`, `.png`, etc.). Las imágenes son cargadas mediante OpenCV.
3. **Inferencia en Cascada (Pipeline IA):**
   * **Paso 1 (Detección):** YOLO escanea la imagen frame a frame para encontrar el patrón de una matrícula (Confidence > 0.70). Se calcula la *bounding box*.
   * **Paso 2 (Recorte):** Se extrae un "zoom" ampliado (*crop*) de la región de interés (ROI).
   * **Paso 3 (Transcripción):** PaddleOCR analiza el recorte en formato RGB y devuelve los caracteres detectados junto con su métrica de confianza.
4. **Postprocesamiento y Storage:** Limpieza de falsos positivos mediante Expresiones Regulares, visualización en el frontend (GUI de OpenCV) y escritura del log de resultados en `/resultados`.

## 🎥 Demostración

https://github.com/user-attachments/assets/3bb6eab0-6063-4977-aa1e-f8fa9da2b000



## ⚠️ Prerrequisitos de Hardware y Modelos

* **Hardware:** Se recomienda encarecidamente disponer de una **GPU compatible con CUDA** (NVIDIA) para acelerar la inferencia multimodelo (YOLO + PaddleOCR). El sistema funcionará en CPU, pero con mayor latencia.
* **Pesos del Modelo:** El script requiere un modelo YOLO previamente entrenado (`yolo11LicensePlateModel.pt`) que se encuentra en el repositorio. 
* **Modelos OCR:** Los modelos predeterminados de sintaxis y detección de PaddleOCR (inglés/universal) se descargarán automáticamente en la primera ejecución si no se encuentran en caché.

## 💻 Instalación (Plug & Play)

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/LicensePlateDetector.git
   cd LicensePlateDetector
   ```

2. **Crea y activa un entorno virtual (recomendado):**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instala las dependencias necesarias:**
   Asegúrate de tener el archivo `requirements.txt` con los paquetes principales (e.g., `opencv-python`, `ultralytics`, `paddleocr`, `paddlepaddle`).
   ```bash
   pip install -r requirements.txt
   ```

4. **Coloca el archivo de pesos (modelo YOLO):**
   Asegúrate de situar el archivo `yolo11LicensePlateModel.pt` en la raíz del proyecto.

## 🎮 Uso del Sistema

1. Crea una carpeta llamada `inputs` en el directorio raíz del proyecto (aunque dejaré en el repositorio mi carpeta de inputs con las imágenes que usé para provar el proyecto):
   ```bash
   mkdir inputs
   ```
   
2. Mueve las fotografías de los vehículos (`.jpg`, `.png`, `.webp`) que deseas analizar dentro de la carpeta `inputs`.
3. Ejecuta el script principal:
   ```bash
   python main.py
   ```
4. **Resultados:** Observa la visualización en tiempo real. Al finalizar presiona `q` o `ESC` para abortar la secuencia temprana. El reporte se guardará automáticamente en el directorio `/resultados`.
