from collections import Counter
import torch
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import os


def get_class_weights(dataset):
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)
    total_samples = sum(class_counts.values())
    
    weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    factor = 2 / sum(weights.values())
    weights = {cls: weight * factor for cls, weight in weights.items()}
    
    weights_list = [weights[i] for i in range(len(weights))]
    return torch.tensor(weights_list, dtype=torch.float)



# def random_90deg_rotation(img):
#     angles = [0, 90, 180, 270]
#     angle = random.choice(angles)
#     return img.rotate(angle)

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    # transforms.CenterCrop((200, 200)),
    # transforms.Lambda(random_90deg_rotation),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])



def plot_images(images:list, display=True, save_at=None, file_name=None):
    n_images = len(images)
    plt.figure(figsize=(5*n_images, 5))
    for i, image in enumerate(images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(image)
        plt.axis('off')
    
    if display:
        plt.show()
        
    if save_at != None:
        plt.savefig(os.path.join(save_at, file_name))
        plt.clf()