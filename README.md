# 🧍‍♂️ Segmentación de Persona y Composición de Fondo con OpenCV

Este proyecto implementa un pipeline completo de procesamiento de imágenes para detectar, segmentar una persona y reemplazar su fondo utilizando técnicas de visión por computadora.

## 🚀 Funcionalidades

### 🔹 Preprocesamiento

* Ajuste de brillo y contraste
* Suavizado con filtro Gaussiano

### 🔹 Segmentación

* Conversión a escala de grises
* Detección de bordes con Canny
* Operaciones morfológicas:

  * Dilatación
  * Cierre (Closing)

### 🔹 Extracción del objeto

* Detección de contornos
* Selección del contorno principal (mayor área)
* Generación de máscara binaria

### 🔹 Composición de imagen

* Extracción de la persona
* Recorte del fondo
* Reemplazo de fondo
* Guardado del resultado final

## 🛠️ Tecnologías utilizadas

* Python
* OpenCV
* NumPy

## ▶️ Cómo ejecutar

1. Clona el repositorio:

```bash id="git4x1"
git clone https://github.com/Asiel-Bee-Bug/person-segmentation-background-composition/tree/main
```

2. Entra a la carpeta:

```bash id="cd4x1"
cd person-segmentation-background-composition
```

3. Instala dependencias:

```bash id="pip4x1"
pip install opencv-python numpy
```

4. Coloca las imágenes necesarias:

```id="img4x1"
persona2.png
ViajeDul.png
```

5. Ejecuta el programa:

```bash id="run4x1"
python person_segmentation.py
```

## 🧠 Flujo del algoritmo

1. Se carga la imagen original
2. Se mejora la imagen (contraste + blur)
3. Se detectan bordes con Canny
4. Se aplican operaciones morfológicas para mejorar la silueta
5. Se obtiene el contorno principal
6. Se genera una máscara binaria
7. Se extrae la persona
8. Se reemplaza el fondo
9. Se guarda la imagen final

## 📌 Notas

* Las imágenes deben estar en la misma carpeta del script
* El resultado final se guarda automáticamente como:

```id="out4x1"
composicion_final.png
```

* Puedes ajustar parámetros como:

  * `alpha` → contraste
  * `umbral_bajo`, `umbral_alto` → Canny
  * tamaño del `kernel`

## 🎯 Objetivo

Aplicar un flujo completo de procesamiento de imágenes para comprender cómo funcionan las técnicas básicas de segmentación y composición en visión por computadora.

