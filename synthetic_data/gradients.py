import numpy as np
from PIL import Image
import math



def apply_gradient(image:Image,
                   brightness_delta:float=50,
                   rotation:float=0):
    """
    Applies a brightness gradient to an image with any rotation angle.
    
    Parameters
    ----------
        image (PIL.Image): Input image.
        brightness_delta (float): Maximum brightness increase.
        rotation (float): Rotation angle in degrees.
        
    Returns
    -------
        PIL.Image: Image with applied gradient.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    x_center = width / 2
    y_center = height / 2
    x_centered = x_coords - x_center
    y_centered = y_coords - y_center
    
    angle_rad = math.radians(rotation)
    # x_rotated = x_centered * math.cos(angle_rad) + y_centered * math.sin(angle_rad)
    y_rotated = -x_centered * math.sin(angle_rad) + y_centered * math.cos(angle_rad)
    rotated_height = abs(height * math.cos(angle_rad)) + abs(width * math.sin(angle_rad))
    
    gradient_values = (y_rotated + rotated_height/2) / rotated_height
    gradient_values = np.clip(gradient_values, 0, 1)
    
    brightness_increase = (gradient_values * brightness_delta).astype(np.uint8)
    gradient = np.stack([brightness_increase, brightness_increase, brightness_increase], axis=2)
    
    image_array = np.array(image)
    result = np.clip(image_array + gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(result)



def apply_vignette(image:Image,
                   strength:int=20,
                   feather:float=0.5,
                   shape:str='circular',
                   offset_x:float=0.0,
                   offset_y:float=0.0):
    """
    Applies a vignette effect to an image.
    
    Parameters
    ----------
        image (PIL.Image): Input image.
        strength (int): Maximum darkness increase (0-255).
        feather (float): Controls how gradual the vignette effect is (0.0-1.0).
                         Lower values create a more defined edge.
        shape (str): 'circular' or 'rectangular' vignette shape.
        offset_x (float): Horizontal offset of vignette center (-1.0 to 1.0).
        offset_y (float): Vertical offset of vignette center (-1.0 to 1.0).
        
    Returns
    -------
        PIL.Image: Image with vignette effect applied.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    width, height = image.size
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    x_center = width / 2 * (1 + offset_x)
    y_center = height / 2 * (1 + offset_y)
    
    x_norm = (x_coords - x_center) / (width / 2)
    y_norm = (y_coords - y_center) / (height / 2)
    
    if shape.lower() == 'circular':
        mask = np.sqrt(x_norm**2 + y_norm**2)
    elif shape.lower() == 'rectangle':
        mask = np.maximum(np.abs(x_norm), np.abs(y_norm))
    

    scale = 1.0 - feather * 0.5
    mask = (mask - scale) / (2.0 - scale)
    mask = np.clip(mask, 0, 1)
    darkening = (mask * strength).astype(np.uint8)
    

    vignette = np.stack([darkening, darkening, darkening], axis=2)
    image_array = np.array(image)
    result = np.clip(image_array - vignette, 0, 255).astype(np.uint8)
    return Image.fromarray(result)