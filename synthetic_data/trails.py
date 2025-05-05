from PIL import Image, ImageDraw
import numpy as np



def add_satellite_trails(
    image, 
    num_trails=1, 
    min_brightness=80, 
    max_brightness=230, 
    min_thickness=0.2, 
    max_thickness=1.0, 
    brightness_gamma=1.5,
    seed=None, 
    antialias_factor=4
):
    """
    Adds multiple random antialiased satellite trails across the image at random positions.
    Brightness and thickness are linked: thinner trails are dimmer.

    Parameters
    ----------
        image (PIL.Image): Input image.
        num_trails (int): Number of satellite trails to add.
        min_brightness (int): Minimum brightness (0-255) for thinnest trail.
        max_brightness (int): Maximum brightness (0-255) for thickest trail.
        min_thickness (float): Minimum trail thickness (in pixels, e.g., 0.2).
        max_thickness (float): Maximum trail thickness (in pixels, e.g., 1.0).
        brightness_gamma (float): Nonlinear scaling factor (>1 = faster dimming for thin).
        seed (int or None): Random seed for reproducibility.
        antialias_factor (int): Supersampling factor for antialiasing (default 4).

    Returns
    -------
        PIL.Image: Image with satellite trails added.
    """
    if seed is not None:
        np.random.seed(seed)

    image = image.convert('RGB')
    width, height = image.size

    highres_w = width * antialias_factor
    highres_h = height * antialias_factor

    trail_layer_highres = Image.new('RGBA', (highres_w, highres_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(trail_layer_highres)

    for _ in range(num_trails):
        thickness = np.random.uniform(min_thickness, max_thickness)
        thickness_norm = (thickness - min_thickness) / (max_thickness - min_thickness)
        thickness_norm = np.clip(thickness_norm, 0, 1)
        thickness_norm_gamma = thickness_norm ** brightness_gamma

        brightness = min_brightness + thickness_norm_gamma * (max_brightness - min_brightness)
        brightness = int(round(np.clip(brightness, 0, 255)))

        cx = np.random.uniform(0, highres_w)
        cy = np.random.uniform(0, highres_h)

        angle = np.random.uniform(0, np.pi)
        diag = np.hypot(highres_w, highres_h) * 1.5

        dx = np.cos(angle) * diag
        dy = np.sin(angle) * diag

        x1 = cx - dx
        y1 = cy - dy
        x2 = cx + dx
        y2 = cy + dy

        highres_thickness = max(int(round(thickness * antialias_factor)), 1)

        draw.line(
            [ (x1, y1), (x2, y2) ], 
            fill=(255, 255, 255, brightness), 
            width=highres_thickness
        )

    trail_layer = trail_layer_highres.resize((width, height), resample=Image.LANCZOS)
    final_image = Image.alpha_composite(image.convert('RGBA'), trail_layer)
    return final_image.convert('RGB')