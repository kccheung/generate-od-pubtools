# in utils.py (or wherever you keep it)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def od_sanity_print(od_hat):
    print("OD matrix shape:", od_hat.shape)
    print("OD matrix (top-left 5x5):")
    print(od_hat[:5, :5])
    print("Min / max OD:", od_hat.min(), od_hat.max())
    print("Total flows:", od_hat.sum())
    print("Zero diagonal? ", (od_hat.diagonal() == 0).all())


def show_regional_image(regional_images, idx, high_res=False):
    """
    regional_images: dict[int, BytesIO or PIL.Image or np.ndarray]
    idx: region index key
    """
    img_obj = regional_images[idx]
    # img_obj = regional_images

    # 1) Convert to a PIL image
    if isinstance(img_obj, BytesIO):
        img_pil = Image.open(img_obj)
        print(img_pil.size)  # (width, height) in pixels, debug use
    elif isinstance(img_obj, Image.Image):
        img_pil = img_obj
    else:
        # maybe already a numpy array
        img_arr = np.array(img_obj)
        plt.imshow(img_arr)
        plt.axis("off")
        plt.title(f"Region {idx}")
        plt.show()
        return

    # 2) Convert to numpy for debugging / plotting
    img_arr = np.array(img_pil)

    # 3) Show
    if high_res:
        plt.figure(figsize=(10, 10))  # bigger window
    plt.imshow(img_arr)
    plt.axis("off")
    plt.title(f"Region {idx}")
    plt.show()
