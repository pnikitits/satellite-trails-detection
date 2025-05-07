import os
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from utils import make_full_image



def ensure_dirs(base_dir='data', splits=('train', 'val', 'test'), classes=('class0', 'class1')):
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)


def generate_image_pair(i, split, base_dir):
    """Generate and save a pair of images (one for each class)."""
    # class0 image (no trails)
    image0 = make_full_image(num_trails=0)
    class0_path = os.path.join(base_dir, split, 'class0', f'image_{i}.jpg')
    image0.save(class0_path)

    # class1 image (with trails)
    image1 = make_full_image(num_trails=1)
    class1_path = os.path.join(base_dir, split, 'class1', f'image_{i}.jpg')
    image1.save(class1_path)
    
    return i


def generate_split_data(split, base_dir, n_samples):
    """Generate all images for a specific data split."""
    num_cores = mp.cpu_count()
    num_workers = max(1, int(num_cores * 0.8))
    
    print(f"Generating {split} set using {num_workers} workers...")
    
    worker_func = partial(generate_image_pair, split=split, base_dir=base_dir)
    
    with mp.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(worker_func, range(n_samples[split])), 
                  total=n_samples[split], 
                  desc=f"Generating {split} set"))
    
    print(f"{split} set generated: {n_samples[split]} samples per class.")



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    base_dir = 'data'
    splits = ['train', 'val', 'test']
    classes = ['class0', 'class1']
    
    n_samples = {
        'train': 20000,
        'val': 4000,
        'test': 4000
    }

    ensure_dirs(base_dir, splits, classes)

    # Process each split sequentially, but parallelize within splits
    for split in splits:
        generate_split_data(split, base_dir, n_samples)

    print("Synthetic dataset generation complete.")