import os
import io
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.pix_safe_cnn import PixSafeCNN

# --- Class Mapping ---
class_names = {0: "nude", 1: "suggestive", 2: "safe"}

# --- Load Model ---
def load_model(model_path, device):
    model = PixSafeCNN(num_classes=3).to(device)
    
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    return model

# --- Preprocess Image ---
def preprocess_image(img, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(img.convert('RGB')).unsqueeze(0)  # Add batch dim

# --- Predict ---
def predict(img_tensor, model, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = probs.argmax()
    return probs, predicted_class

# --- Load Image from Path or URL ---
def load_image(image_path=None, image_url=None):
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ File not found: {image_path}")
        print(f"📁 Loading local image: {image_path}")
        return Image.open(image_path)

    elif image_url:
        print(f"🌐 Downloading image from URL: {image_url}")
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"❌ Failed to download or open image: {e}")

    else:
        raise ValueError("❌ Provide either a valid image_path or image_url.")

# --- Main Entry ---
if __name__ == "__main__":
    # 👇 Set one of these:
    image_path = None  # e.g., "your_image.jpg"
    image_url = "https://example.com/sample.jpg"

    model_path = "models/pixsafe_cnn.safetensors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    model = load_model(model_path, device)

    image = load_image(image_path=image_path, image_url=image_url)
    img_tensor = preprocess_image(image)
    probs, pred_class = predict(img_tensor, model, device)

    print("\n🔍 Inference Result")
    print(f"→ Prediction probabilities: {probs}")
    print(f"→ Predicted class: {pred_class} ({class_names[pred_class]})")
