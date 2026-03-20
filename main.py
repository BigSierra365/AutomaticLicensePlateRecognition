"""
SISTEMA DE RECONOCIMIENTO AUTOMÁTICO DE MATRÍCULAS (ALPR) - PROCESAMIENTO POR IMÁGENES

Propósito:
Este script permite procesar una serie de imágenes estáticas recolectadas en una carpeta para identificar
matrículas vehiculares, extraer su texto de forma automática y generar un informe de resultados.

Librerías principales:
- YOLO (Ultralytics): Utilizada para la detección del objeto 'matrícula' dentro de la imagen.
- PaddleOCR: Motor de reconocimiento óptico de caracteres especializado en lectura de texto.
- OpenCV/imutils: Herramientas para la lectura, redimensión y visualización de las imágenes.
- Re (Expresiones Regulares): Para el filtrado y limpieza del texto detectado.

Flujo de ejecución:
1. Carga de modelos: Se inicializan los pesos de la red neuronal YOLO y el motor OCR.
2. Escaneo de archivos: Se listan todas las imágenes compatibles en la carpeta de entrada.
3. Bucle de detección: Para cada imagen, se localizan las matrículas y se extrae el texto.
4. Informe final: Se muestra una tabla resumen en consola y se exporta a un archivo de texto.
"""

import cv2
import imutils
import re
import glob
import os
import time 
import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# 1. CARGAMOS LOS MODELOS DE INTELIGENCIA ARTIFICIAL UNA SOLA VEZ
# Se cargan los modelos al inicio para evitar esperas durante el procesamiento de imágenes
print("Cargando modelos de IA...")
model = YOLO("yolo11LicensePlateModel.pt")
# El motor OCR detecta la orientación del texto para mejorar la precisión de lectura
ocr = PaddleOCR(use_textline_orientation=True, lang='en') 
print("Modelos cargados con éxito.\n")

# 2. LOCALIZACIÓN DE TODAS LAS IMÁGENES DISPONIBLES
# Definimos la carpeta de entrada y los formatos de imagen compatibles para el análisis
ruta_carpeta = r".\inputs"
tipos_archivos = ('*.jpg', '*.jpeg', '*.png', '*.webp')
rutas_imagenes = []
for extension in tipos_archivos:
    rutas_imagenes.extend(glob.glob(os.path.join(ruta_carpeta, extension)))

# Estructura de datos para almacenar la información recolectada de cada matrícula
informe_resultados = []

# Verificación de existencia de archivos en el directorio especificado
if not rutas_imagenes:
    print(f"No se encontraron imágenes en {ruta_carpeta}")
else:
    print(f"Se han encontrado {len(rutas_imagenes)} imágenes. Comenzando análisis automático...\n")

# 3. PROCESAMIENTO AUTOMÁTICO IMAGEN POR IMAGEN
# Iniciamos el recorrido por cada una de las rutas de archivos encontradas
for img_path in rutas_imagenes:
    nombre_archivo = os.path.basename(img_path)
    print("-" * 50)
    print(f"Analizando: {nombre_archivo} ...")
    
    # Control de tiempo para medir cuánto tarda el sistema en analizar cada fotografía
    tiempo_inicio = time.time() 
    
    # Intento de lectura de la imagen desde disco
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error al leer la imagen {nombre_archivo}. Saltando...")
        continue

    # Ejecución del modelo de visión artificial YOLO para localizar el recuadro de la matrícula
    results = model(image)
    
    # Buffer temporal para gestionar múltiples matrículas encontradas en una sola imagen
    matriculas_en_esta_imagen = []

    # Interpretación de los resultados obtenidos por el buscador de objetos
    for result in results:
        # Filtramos las detecciones para quedarnos solo con lo que el modelo clasifica como placa
        index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]

        for idx in index_plates:
            conf_caja = result.boxes.conf[idx].item()
            # Solo procesamos detecciones donde la IA esté razonablemente segura de que es una matrícula
            if conf_caja > 0.7:
                xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                
                # Definición del área de recorte añadiendo un pequeño margen alrededor
                h_img, w_img, _ = image.shape
                y_start, y_end = max(0, y1-10), min(h_img, y2+10)
                x_start, x_end = max(0, x1-10), min(w_img, x2+10)
                
                # Extracción del zoom dedicado únicamente a la matrícula
                plate_image = image[y_start:y_end, x_start:x_end]

                # Si el recorte es válido, procedemos a realizar la lectura del texto (OCR)
                if plate_image.size > 0:
                    # Convertimos a formato de color compatible con el motor de lectura
                    result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
                    
                    if result_ocr and result_ocr[0]:
                        boxes = result_ocr[0]['rec_boxes']
                        texts = result_ocr[0]['rec_texts']
                        scores = result_ocr[0]['rec_scores'] 
                        
                        # Organización de los fragmentos de texto de izquierda a derecha
                        left_to_right = sorted(zip(boxes, texts, scores), key=lambda x: min(x[0][::2])) 

                        # Limpieza del texto: eliminamos caracteres especiales y aplicamos filtros básicos
                        whitelist_pattern = re.compile(r'^[A-Z0-9]+$')
                        text_joined = ''.join([t for _, t, _ in left_to_right])
                        output_text = ''.join([t for t in text_joined if whitelist_pattern.fullmatch(t)])
                        
                        # Cálculo de la fiabilidad media de la lectura realizada por el motor OCR
                        confianza_final = 0.0
                        if scores:
                            confianza_final = sum(scores) / len(scores)

                        # Representación de resultados en terminal
                        if output_text:
                            print(f"   --> Matrícula: {output_text} | Confianza OCR: {confianza_final:.2f}")
                            
                            # Almacenamiento de los datos de la matrícula en la memoria temporal de la imagen actual
                            matriculas_en_esta_imagen.append({
                                "texto": output_text,
                                "confianza": confianza_final
                            })

                        # Proceso visual para dibujar el recuadro verde y el texto sobre la imagen original
                        cv2.imshow("Plate Recorte", plate_image)

                        # Cambiar modo de ventana para que la ventana se abra en primer plano
                        cv2.setWindowProperty("Plate Recorte", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.setWindowProperty("Plate Recorte", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)      
                        
                        cv2.rectangle(image, (x1 -10, y1 -35), (x2 +10, y2-(y2 -y1)), (0, 255, 0), -1)
                        cv2.rectangle(image, (x1 , y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, output_text, (x1 -7, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Cálculo final del tiempo de respuesta del sistema para esta fotografía
    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio

    # Integración de los hallazgos de esta imagen en la lista general para el reporte final
    if len(matriculas_en_esta_imagen) == 0:
        # Registro en caso de no haberse detectado ningún elemento en el archivo actual
        informe_resultados.append({
            "archivo": nombre_archivo,
            "matricula": "No detectada",
            "confianza": "0.0%",
            "tiempo": f"{tiempo_total:.2f}s"
        })
    else:
        # Registro detallado para cada una de las placas encontradas en la escena
        for placa in matriculas_en_esta_imagen:
            informe_resultados.append({
                "archivo": nombre_archivo,
                "matricula": placa["texto"],
                "confianza": f"{placa['confianza']*100:.1f}%",
                "tiempo": f"{tiempo_total:.2f}s"
            })

    # Visualización del proceso en tiempo real para el usuario
    cv2.imshow("Image Analizada", imutils.resize(image, width=800))
    # Cambiar modo de ventana para que la ventana se abra en primer plano
    cv2.setWindowProperty("Image Analizada", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Image Analizada", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    # Pausa programada para permitir la visualización antes de pasar al siguiente archivo
    tecla = cv2.waitKey(1500) & 0xFF 
    if tecla == ord('q') or tecla == 27:
        print("\n[!] Secuencia interrumpida por el usuario.")
        break
    
    # Limpieza de ventanas antes de iniciar un nuevo análisis
    cv2.destroyAllWindows()

# 4. GENERACIÓN DE INFORMES FINALES Y EXPORTACIÓN A ARCHIVO
# Cierre final de cualquier ventana residual de procesamiento visual
cv2.destroyAllWindows()

# Construcción de la tabla resumen para visualización en consola
lineas_informe = []
lineas_informe.append("\n" + "="*60)
lineas_informe.append(" "*15 + "INFORME FINAL DE ANÁLISIS")
lineas_informe.append("="*60)
lineas_informe.append(f"{'ARCHIVO':<20} | {'MATRÍCULA':<15} | {'CONFIANZA':<10} | {'TIEMPO'}")
lineas_informe.append("-" * 60)

for res in informe_resultados:
    lineas_informe.append(f"{res['archivo']:<20} | {res['matricula']:<15} | {res['confianza']:<10} | {res['tiempo']}")

lineas_informe.append("="*60)
lineas_informe.append("Proceso finalizado con éxito.\n")

# Conversión de la lista de strings en un bloque de texto unificado
texto_final = "\n".join(lineas_informe)

# Presentación del reporte detallado por terminal
print(texto_final)

# Guardado permanente del informe en un archivo de texto dentro de la subcarpeta de resultados
# Garantizamos la creación automática de la carpeta si esta no existe
os.makedirs("resultados", exist_ok=True)

# Generación de una estampa de tiempo única para el nombre del archivo de salida
fecha_hora_actual = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
nombre_archivo_txt = f"res-{fecha_hora_actual}.txt"
ruta_completa_txt = os.path.join("resultados", nombre_archivo_txt)

# Escritura física del archivo de reporte en disco
with open(ruta_completa_txt, "w", encoding="utf-8") as archivo:
    archivo.write(texto_final)

print(f"El informe se ha guardado correctamente en: {ruta_completa_txt}")