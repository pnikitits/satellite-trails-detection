import numpy as np
from PIL import Image, ImageDraw, ImageFilter



def add_soft_stars(
    image,
    num_stars=50,
    min_size=0.1,
    max_size=1,
    seed=None,
    max_brightness=255,
    min_brightness=50
):
    """
    Adds soft, glowing stars to an image — sharper and size-scaled brightness.

    Parameters
    ----------
        image (PIL.Image): The input image to modify.
        num_stars (int): Number of stars to add.
        min_size (float): Minimum star radius (can be <1).
        max_size (float): Maximum star radius.
        seed (int or None): Random seed for reproducibility.
        max_brightness (int): Maximum brightness of the stars (default 255).

    Returns
    -------
        PIL.Image: Image with soft stars added.
    """
    if seed is not None:
        np.random.seed(seed)

    image = image.convert('RGB')
    width, height = image.size
    star_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    for _ in range(num_stars):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.uniform(min_size, max_size)

        size_scale = (radius - min_size) / (max_size - min_size + 1e-5)
        brightness = int(np.clip(size_scale * max_brightness, min_brightness, max_brightness))

        star_size = int(np.ceil(radius * 4))
        if star_size < 3:
            star_size = 3

        star = Image.new('L', (star_size, star_size), 0)
        draw = ImageDraw.Draw(star)

        center = star_size // 2
        r = max(radius, 0.5)

        draw.ellipse(
            [ (center - r, center - r), (center + r, center + r) ],
            fill=brightness
        )

        blur_amount = max(radius * 0.5, 0.3)
        blurred_star = star.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        star_layer.paste((255, 255, 255, brightness), (x - center, y - center), mask=blurred_star)

    final_image = Image.alpha_composite(image.convert('RGBA'), star_layer)
    return final_image.convert('RGB')