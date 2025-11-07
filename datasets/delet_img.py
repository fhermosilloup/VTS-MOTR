import os
import glob

# Ruta raíz del dataset
root_dir = "DETRAC-MOT"

# Subcarpetas donde buscar
splits = ["train", "val"]

# Extensiones de imagen a eliminar
extensions = (".jpg", ".jpeg", ".png")

for split in splits:
    split_path = os.path.join(root_dir, split)
    if not os.path.exists(split_path):
        print(f"Directorio no encontrado: {split_path}")
        continue

    # Recorre todas las carpetas MVI_XXXXX/img
    for mvi_dir in os.listdir(split_path):
        img_dir = os.path.join(split_path, mvi_dir, "img")

        if not os.path.isdir(img_dir):
            continue

        # Buscar imágenes en la carpeta img
        images = glob.glob(os.path.join(img_dir, "*"))
        deleted = 0

        for img_path in images:
            if img_path.lower().endswith(extensions):
                try:
                    os.remove(img_path)
                    deleted += 1
                except Exception as e:
                    print(f"Error eliminando {img_path}: {e}")

        print(f"Eliminadas {deleted} imágenes en {img_dir}")

print("Proceso completado ✅")



