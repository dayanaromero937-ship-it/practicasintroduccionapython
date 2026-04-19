import os
import numpy as np
from PIL import Image

def preparar_imagen(archivo):
    """Función pura para limpieza y preparación (Fase A)."""
    # Usamos la ruta donde subiste tus fotos
    ruta = os.path.join('data/raw', archivo)
    # Convertimos a escala de grises y tamaño 28x28
    img = Image.open(ruta).convert('L').resize((28, 28))
    return np.array(img)

def red_neuronal_funcional(X, pesos, sesgos):
    """Inferencia mediante producto punto (Fase B)."""
    # Operación vectorial: Z = X · W + b
    z = np.dot(X, pesos) + sesgos
    # Activación Softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)