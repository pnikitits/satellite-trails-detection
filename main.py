import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from model.inference import predict_image
from model.cnn import CNN_V1
from model.train import train_model, evaluate_model
from model.utils import get_class_weights, transform, plot_images
from model.plots import plot_metrics



def prepare_dirs(runs_dir='runs'):
    os.makedirs(runs_dir, exist_ok=True)
    current_run_name = f"run_{len(os.listdir(runs_dir)) + 1} {time.strftime('%Y%m%d_%H%M%S')}"
    current_run_dir = os.path.join(runs_dir, current_run_name)
    os.makedirs(current_run_dir, exist_ok=True)
    return current_run_dir
    

def main():
    current_run_dir = prepare_dirs()
    
    device = "mps" if torch.mps.is_available() else "cpu"
    
    train_dir = "synthetic_data/data/train"
    val_dir = "synthetic_data/data/val"
    test_dir = "synthetic_data/data/test"
    real_dir = "real_data"
    
    batch_size = 64

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    real_dataset = datasets.ImageFolder(real_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    
    class_weights = get_class_weights(train_dataset).to(device)
    
    model = CNN_V1().to(device)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    
    history = train_model(model=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          real_loader=real_loader,
                          criterion=criterion,
                          optimiser=optimiser,
                          device=device,
                          epochs=10,
                          verbose=False,
                          return_history=True)

    # save the history as plot and json
    history_path = os.path.join(current_run_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    plot_metrics(history, display=False, save_at=current_run_dir)
    
    
    # test on fake data
    pred_1, prob_1, overlayed_1 = predict_image(model, 'synthetic_data/data/test/class1/image_80.jpg', transform, device)
    pred_2, prob_2, overlayed_2 = predict_image(model, 'synthetic_data/data/test/class0/image_80.jpg', transform, device)
    
    plot_images([overlayed_1, overlayed_2],
                display=False,
                save_at=current_run_dir,
                file_name="fake_test.png")
    
    # test on real data
    pred_1, prob_1, overlayed_1 = predict_image(model, 'real_data/class1/Light_M63_120.0s_Bin1_20250427-235057_0021_thn.jpg', transform, device, verbose=False)
    pred_2, prob_2, overlayed_2 = predict_image(model, 'real_data/class1/Light_M94_120.0s_Bin1_20250430-224434_0014_thn.jpg', transform, device, verbose=False)
    pred_3, prob_3, overlayed_3 = predict_image(model, 'real_data/class1/Light_M81_120.0s_Bin1_20250304-204937_0022_thn.jpg', transform, device, verbose=False)

    pred_4, prob_4, overlayed_4 = predict_image(model, 'real_data/class0/Light_M63_120.0s_Bin1_20250428-011151_0003_thn.jpg', transform, device, verbose=False)
    pred_5, prob_5, overlayed_5 = predict_image(model, 'real_data/class0/Light_M94_120.0s_Bin1_20250501-003332_0007_thn.jpg', transform, device, verbose=False)
    pred_6, prob_6, overlayed_6 = predict_image(model, 'real_data/class0/Light_NGC4565_120.0s_Bin1_20250305-052744_0070_thn.jpg', transform, device, verbose=False)

    plot_images([overlayed_1, overlayed_2, overlayed_3],
                display=False,
                save_at=current_run_dir,
                file_name="real_test_class_1")

    plot_images([overlayed_4, overlayed_5, overlayed_6],
                display=False,
                save_at=current_run_dir,
                file_name="real_test_class_0")
    


if __name__ == "__main__":
    main()