from utils import generate_noise_image
from stars import add_soft_stars
from trails import add_satellite_trails
import random
import os


def make_full_image(num_trails):
    image = generate_noise_image(max_brightness=50)
    image = add_soft_stars(image, num_stars=random.randint(50, 100))
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
        'train': 100,
        'val': 20,
        'test': 30
    }

    ensure_dirs(base_dir, splits, classes)

    for split in splits:
        for i in range(n_samples[split]):
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