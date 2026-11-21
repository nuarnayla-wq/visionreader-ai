from pathlib import Path
import os
import torch
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path("models/custom_mobilenetv3.pth")
PREPROCESS = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class ImageReader:
    """CPU-only MobileNetV3 image classifier with optional custom weights."""
    def __init__(self, custom_model_path=MODEL_PATH):
        path = Path(custom_model_path)
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu")
            self.labels = checkpoint["classes"]
            self.model = models.mobilenet_v3_small(weights=None)
            self.model.classifier[3] = torch.nn.Linear(
                self.model.classifier[3].in_features, len(self.labels))
            self.model.load_state_dict(checkpoint["state_dict"])
            self.custom = True
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
            self.labels = weights.meta["categories"]
            self.custom = False
        self.model.cpu().eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k=5):
        tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0)
        probs = torch.softmax(self.model(tensor)[0], dim=0)
        scores, ids = torch.topk(probs, min(top_k, len(self.labels)))
        return [{"label": self.labels[int(i)], "confidence": float(s)}
                for s, i in zip(scores, ids)]

def read_text(image: Image.Image):
    """OCR text in Indonesian and English when Tesseract is installed."""
    try:
        import pytesseract
        command = os.environ.get("TESSERACT_CMD")
        if command:
            pytesseract.pytesseract.tesseract_cmd = command
        return pytesseract.image_to_string(image, lang="ind+eng").strip()
    except Exception as exc:
        return f"OCR belum tersedia: {exc}"
