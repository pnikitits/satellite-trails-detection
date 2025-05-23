import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from stars import add_soft_stars
from trails import add_satellite_trails
from gradients import apply_gradient, apply_vignette


def make_full_image(num_trails):
    image = generate_noise_image(mean_brightness=random.randint(30, 40), noise_std=random.randint(3, 10))
    image = add_soft_stars(image, num_stars=random.randint(60, 130), min_brightness=10, max_brightness=255)
    
    for _ in range(2):
        image = apply_vignette(image, strength=random.randint(10, 20), offset_x=random.uniform(-0.2, 0.2), offset_y=random.uniform(-0.2, 0.2), feather=random.uniform(0.5, 0.9))
    
    for _ in range(4):
        image = apply_gradient(image, brightness_delta=random.randint(0, 10), rotation=random.uniform(0, 359))
        
    image = add_satellite_trails(image, num_trails=num_trails)
    return image



def generate_noise_image(width=200, height=200, mean_brightness=10, noise_std=3):
    noise = np.random.normal(loc=mean_brightness, scale=noise_std, size=(height, width, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(noise, 'RGB')
    return image



def sample_average_color(image, grid_points=20):
    image = image.convert('RGB')
    width, height = image.size
    
    grid_size = int(np.ceil(np.sqrt(grid_points)))
    x_coords = np.linspace(0, width - 1, grid_size, dtype=int)
    y_coords = np.linspace(0, height - 1, grid_size, dtype=int)
    
    sampled_colors = []
    for x in x_coords:
        for y in y_coords:
            color = image.getpixel((x, y))
            sampled_colors.append(color)
            if len(sampled_colors) >= grid_points:
                break
        if len(sampled_colors) >= grid_points:
            break
    
    sampled_array = np.array(sampled_colors, dtype=np.float32)
    
    min_color = sampled_array.min(axis=0)
    avg_color = sampled_array.mean(axis=0)
    max_color = sampled_array.max(axis=0)
    
    min_value = np.mean(min_color)
    avg_value = np.mean(avg_color)
    max_value = np.mean(max_color)
    
    return min_value, avg_value, max_value



def load_image(image_path):
    image = Image.open(image_path)
    
    image_array = np.array(image)
    print("Numpy dtype:", image_array.dtype)
    
    return image


def plot_images(images:list):
    n_images = len(images)
    plt.figure(figsize=(5*n_images, 5))
    for i, image in enumerate(images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()