import cv2

def convertir_grises(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def ajustar_brillo(imagen, alpha: float = 1.2, beta: int = 30):
    return cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)
