from utils import generate_noise_image
from stars import add_soft_stars
from trails import add_satellite_trails
from gradients import apply_gradient, apply_vignette
import random
import os
from tqdm import tqdm


def make_full_image(num_trails):
    image = generate_noise_image(mean_brightness=43, noise_std=5)
    image = apply_vignette(image, strength=random.randint(5, 20), offset_x=random.uniform(-0.2, 0.2), offset_y=random.uniform(-0.2, 0.2))
    image = add_soft_stars(image, num_stars=random.randint(50, 100))
    image = apply_gradient(image, brightness_delta=random.randint(0, 15), rotation=random.uniform(0, 359))
    return add_satellite_trails(image, num_trails=num_trails)


def ensure_dirs(base_dir='data', splits=('train', 'val', 'test'), classes=('class0', 'class1')):
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)



if __name__ == "__main__":
    base_dir = 'data'
    splits = ['train', 'val', 'test']
    classes = ['class0', 'class1']
    
    n_samples = {
        'train': 6000,
        'val': 2000,
        'test': 2000
    }

    ensure_dirs(base_dir, splits, classes)

    for split in splits:
        for i in tqdm(range(n_samples[split])):
            # class0 image (no trails)
            image0 = make_full_image(num_trails=0)
            class0_path = os.path.join(base_dir, split, 'class0', f'image_{i}.jpg')
            image0.save(class0_path)

            # class1 image (with trails)
            image1 = make_full_image(num_trails=1)
            class1_path = os.path.join(base_dir, split, 'class1', f'image_{i}.jpg')
            image1.save(class1_path)

        print(f"{split} set generated: {n_samples[split]} samples per class.")

    print("Synthetic dataset generation complete.")