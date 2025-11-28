# in utils.py (or wherever you keep it)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def show_regional_image(regional_images, idx):
    """
    regional_images: dict[int, BytesIO or PIL.Image or np.ndarray]
    idx: region index key
    """
    img_obj = regional_images[idx]
    # img_obj = regional_images

    # 1) Convert to a PIL image
    if isinstance(img_obj, BytesIO):
        img_pil = Image.open(img_obj)
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
    plt.imshow(img_arr)
    plt.axis("off")
    plt.title(f"Region {idx}")
    plt.show()
