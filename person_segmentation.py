import cv2
import numpy as np

# PUNTO 1: CARGA DE LA IMAGEN

# Cargar la imagen original desde archivo
imagen_original = cv2.imread("persona2.png")

# Verificar que la imagen se cargó correctamente
if imagen_original is None:
    print("Error: No se pudo cargar la imagen.")
else:
    print(f"Imagen cargada correctamente.")
    print(f"Dimensiones: {imagen_original.shape[1]}px ancho x {imagen_original.shape[0]}px alto")
    print(f"Canales: {imagen_original.shape[2]}")

    # Mostrar la imagen original
    cv2.imshow("Punto 1 - Imagen Original", imagen_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    




# PUNTO 2: PREPROCESAMIENTO DE LA IMAGEN

# 1: Ajuste de brillo y contraste 
# alpha: controla el contraste (1.0 = sin cambio, >1.0 = más contraste)
# beta:  controla el brillo  (0 = sin cambio, valores positivos = más brillo)
alpha = 1.2  # Contraste
beta = 0    # Brillo

imagen_contraste = cv2.convertScaleAbs(imagen_original, alpha=alpha, beta=beta)

# Mostrar resultado del ajuste de brillo/contraste
cv2.imshow("Punto 2a - Brillo y Contraste", imagen_contraste)
cv2.waitKey(0)

# 2: Blur Gaussiano 
# Se aplica sobre la imagen con contraste ajustado
# (5, 5): tamaño del kernel (debe ser impar)
# 0: desviación estándar calculada automáticamente por OpenCV
imagen_blur = cv2.GaussianBlur(imagen_contraste, (5, 5), 0)

# Mostrar resultado del blur
cv2.imshow("Punto 2b - Blur Gaussiano", imagen_blur)
cv2.waitKey(0)

cv2.destroyAllWindows()





# PUNTO 3: SEGMENTACIÓN CON DETECCIÓN DE BORDES (CANNY)

# Canny requiere una imagen en escala de grises
# Se convierte la imagen preprocesada (blur) para mantener la mejora aplicada
imagen_gris = cv2.cvtColor(imagen_blur, cv2.COLOR_BGR2GRAY)

# Mostrar imagen en escala de grises
cv2.imshow("Punto 3a - Escala de Grises", imagen_gris)
cv2.waitKey(0)

# --- Detección de bordes con Canny 
# umbral_bajo:  píxeles con gradiente menor a este valor se descartan
# umbral_alto:  píxeles con gradiente mayor a este valor se aceptan como borde
# Entre ambos umbrales: se aceptan solo si están conectados a un borde fuerte
umbral_bajo = 4
umbral_alto = 20

imagen_canny = cv2.Canny(imagen_gris, umbral_bajo, umbral_alto)

# Mostrar resultado de Canny
cv2.imshow("Punto 3b - Deteccion de Bordes Canny", imagen_canny)
cv2.waitKey(0)

cv2.destroyAllWindows()





# PUNTO 4: OPERACIONES MORFOLÓGICAS

# El elemento estructurante define la forma y tamaño de la operación morfológica
# MORPH_RECT: kernel rectangular
# (5, 5): tamaño del kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 1: Dilatación 
# Engrosa los bordes detectados por Canny
# Ayuda a conectar bordes que quedaron fragmentados (especialmente en el cabello)
imagen_dilatada = cv2.dilate(imagen_canny, kernel)

# Mostrar resultado de la dilatación
cv2.imshow("Punto 4a - Dilatacion", imagen_dilatada)
cv2.waitKey(0)

# 2: Cierre (Closing) 
# Combina Dilatación + Erosión en una sola operación
# Rellena huecos pequeños dentro del contorno de la persona
# Es ideal aplicarlo después de la dilatación para consolidar la silueta
imagen_cierre = cv2.morphologyEx(imagen_dilatada, cv2.MORPH_CLOSE, kernel)

# Mostrar resultado del cierre
cv2.imshow("Punto 4b - Cierre (Closing)", imagen_cierre)
cv2.waitKey(0)

cv2.destroyAllWindows()





# PUNTO 5: GENERACIÓN DE LA MÁSCARA BINARIA

# Buscar todos los contornos presentes en la imagen después de las operaciones morfológicas
# RETR_EXTERNAL: recupera únicamente los contornos externos (descarta contornos internos)
# CHAIN_APPROX_SIMPLE: comprime segmentos horizontales, verticales y diagonales
contornos, _ = cv2.findContours(imagen_cierre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Seleccionar el contorno de mayor área (corresponde a la silueta de la persona)
contorno_principal = max(contornos, key=cv2.contourArea)

# Crear una imagen negra del mismo tamaño que la original (fondo de la máscara)
mascara = np.zeros(imagen_original.shape[:2], dtype=np.uint8)

# Dibujar el contorno principal relleno en blanco sobre la máscara negra
# -1 en thickness indica que el contorno se dibuja completamente relleno
# 255: color blanco (región de interés)

cv2.drawContours(mascara, [contorno_principal], -1, 255, thickness=cv2.FILLED)

# Mostrar la máscara binaria generada
cv2.imshow("Punto 5 - Mascara Binaria", mascara)
cv2.waitKey(0)

cv2.destroyAllWindows()





# PUNTO 6: APLICACIÓN DE LA MÁSCARA Y COMPOSICIÓN FINAL

# Paso 1: Extraer la persona usando la máscara ---
# bitwise_and aplica la máscara sobre la imagen original
# Solo conserva los píxeles donde la máscara es blanca (255)
persona_extraida = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara)

# Mostrar la persona extraída sobre fondo negro
cv2.imshow("Punto 6a - Persona Extraida", persona_extraida)
cv2.waitKey(0)

# Paso 2: Cargar imagen de fondo ---
fondo = cv2.imread("ViajeF.png")

# Verificar que el fondo se cargó correctamente
if fondo is None:
    print("Error: No se pudo cargar la imagen de fondo.")
else:
    # Redimensionar el fondo al mismo tamaño que la imagen original
    # Garantiza que ambas imágenes sean compatibles para la composición
    fondo = cv2.resize(fondo, (imagen_original.shape[1], imagen_original.shape[0]))

    # --- Paso 3: Invertir la máscara para obtener la región del fondo ---
    # La máscara invertida tiene blanco donde estaba el fondo original
    # Esto nos permite "recortar" el fondo nuevo en la zona donde no está la persona
    mascara_invertida = cv2.bitwise_not(mascara)
    fondo_recortado = cv2.bitwise_and(fondo, fondo, mask=mascara_invertida)

    # Mostrar el fondo recortado (sin la región de la persona)
    cv2.imshow("Punto 6b - Fondo Recortado", fondo_recortado)
    cv2.waitKey(0)

    # --- Paso 4: Combinar persona extraída + fondo nuevo ---
    # cv2.add() suma ambas imágenes píxel a píxel
    # Como son complementarias (una tiene negro donde la otra tiene contenido),
    # el resultado es la composición final sin solapamiento
    composicion_final = cv2.add(persona_extraida, fondo_recortado)

    # Mostrar composición final
    cv2.imshow("Punto 6c - Composicion Final", composicion_final)
    cv2.waitKey(0)

    # Guardar la composición final en disco
    cv2.imwrite("composicion_final.png", composicion_final)
    print("Composicion final guardada como: composicion_final.png")

cv2.destroyAllWindows()
