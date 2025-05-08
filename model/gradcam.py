import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2



def generate_gradcam(model, image_tensor, device, target_class=None):
    model.eval()
    gradients = []
    activations = []
    
    # for name, module in model.named_modules():
    #     print(name, module)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Last conv layer (this is model specific)
    target_layer = model.features[-3]  
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor).squeeze()

    if target_class is None:
        target_class = (torch.sigmoid(output) > 0.5).int()

    model.zero_grad()
    output.backward()

    grads = gradients[0].detach().cpu().numpy()[0]  
    acts = activations[0].detach().cpu().numpy()[0]  

    weights = np.mean(grads, axis=(1, 2))  # (C,)
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (200, 200))

    fwd_handle.remove()
    bwd_handle.remove()

    return cam


def plot_gradcam_on_image(image_tensor, cam, alpha=0.4):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 0.5) + 0.5

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    overlayed = alpha * heatmap + (1 - alpha) * image_np
    overlayed = overlayed / np.max(overlayed)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(overlayed)
    # plt.axis('off')
    # plt.show()
    return overlayed