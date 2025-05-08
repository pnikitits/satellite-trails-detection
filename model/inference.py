from PIL import Image
import torch
from gradcam import generate_gradcam, plot_gradcam_on_image



def predict_image(model, image_path, transform, device, show_gradcam=True, verbose=False):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    image_tensor = image_tensor.to(device)
    image_tensor_batch = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor_batch)
        probs = torch.sigmoid(outputs).item()
        pred = int(probs > 0.5)

    print(f'Prediction: {pred} | Probability: {probs:.4f}') if verbose else None

    if show_gradcam:
        cam = generate_gradcam(model, image_tensor, device)
        overlayed = plot_gradcam_on_image(image_tensor.cpu(), cam)
        return pred, probs, overlayed

    return pred, probs, None