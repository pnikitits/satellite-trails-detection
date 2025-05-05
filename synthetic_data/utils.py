import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def generate_noise_image(width=200, height=200, min_brightness=0, max_brightness=255):
    random_data = np.random.randint(min_brightness, max_brightness, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(random_data, 'RGB')
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
    avg_color = sampled_array.mean(axis=0)
    return tuple(int(round(c)) for c in avg_color)



def load_image(image_path):
    image = Image.open(image_path)
    return image


def plot_images(images:list):
    n_images = len(images)
    plt.figure(figsize=(5*n_images, 5))
    for i, image in enumerate(images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()