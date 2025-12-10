# s0_2_gif_gen.py
import numpy as np
import imageio.v2 as imageio
from PIL import Image


def load_square_image(path, size=1150, bg_color=(255, 255, 255, 255)):
    """
    Load an image, resize it to fit inside a size x size square while
    preserving aspect ratio, and pad with background to exactly size x size.
    """
    im = Image.open(path).convert("RGBA")

    # Resize to fit within the square, preserving aspect ratio
    im.thumbnail((size, size), Image.LANCZOS)

    # Create square canvas
    canvas = Image.new("RGBA", (size, size), bg_color)

    # Center the resized image on the canvas
    x = (size - im.width) // 2
    y = (size - im.height) // 2
    canvas.paste(im, (x, y), im)

    return np.array(canvas)


def make_gif(image_paths, out_path, seconds_per_frame=0.5, size=1150):
    """
    image_paths: list of file paths (in order)
    out_path: output .gif path, e.g. 'out.gif'
    seconds_per_frame: duration of each frame
    size: final width/height in pixels (square)
    """
    frames = []
    for path in image_paths:
        frame = load_square_image(path, size=size)
        frames.append(frame)

    imageio.mimsave(out_path, frames, duration=seconds_per_frame)
    print(f"Saved GIF to {out_path}")


if __name__ == "__main__":
    # example usage
    imgs = [
        "./docs/img/fukuoka_ego_origin_inflow_0.png",
        "./docs/img/fukuoka_ego_origin_outflow_0.png",
        "./docs/img/fukuoka_ego_origin_inflow_1.png",
        "./docs/img/fukuoka_ego_origin_outflow_1.png",
        "./docs/img/fukuoka_ego_origin_inflow_2.png",
        "./docs/img/fukuoka_ego_origin_outflow_2.png",
        "./docs/img/fukuoka_ego_origin_inflow_3.png",
        "./docs/img/fukuoka_ego_origin_outflow_3.png",
        "./docs/img/fukuoka_ego_origin_inflow_4.png",
        "./docs/img/fukuoka_ego_origin_outflow_4.png",
    ]
    make_gif(imgs, "./docs/img/fukuoka_ego_demo.gif", seconds_per_frame=5000, size=1150)
