from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from image_processing.loader import cargar_imagen
from image_processing.transform import convertir_grises, ajustar_brillo

def main():
    ruta = Path(__file__).resolve().parent.parent.parent / "data" / "goku.jpg"
    imagen = cargar_imagen(str(ruta))

    if imagen is None:
        raise FileNotFoundError("No se pudo cargar la imagen")

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    gris = convertir_grises(imagen)
    brillo = ajustar_brillo(gris, alpha=1.2, beta=30)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(imagen_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gris, cmap="gray")
    plt.title("Grises")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(brillo, cmap="gray")
    plt.title("Brillo")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()