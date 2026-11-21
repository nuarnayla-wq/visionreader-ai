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


# Lightweight MobileNet-SSD detector (VOC classes), separate from image classification.
DETECTION_DIR = Path("models/detection")
PROTOTXT_PATH = DETECTION_DIR / "MobileNetSSD_deploy.prototxt"
WEIGHTS_PATH = DETECTION_DIR / "mobilenet_iter_73000.caffemodel"
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt"
WEIGHTS_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
VOC_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "dining table", "dog",
    "horse", "motorbike", "person", "potted plant", "sheep", "sofa",
    "train", "tv/monitor",
]
ID_LABELS = {
    "aeroplane": "pesawat", "bicycle": "sepeda", "bird": "burung",
    "boat": "perahu", "bottle": "botol", "bus": "bus", "car": "mobil",
    "cat": "kucing", "chair": "kursi", "cow": "sapi",
    "dining table": "meja makan", "dog": "anjing", "horse": "kuda",
    "motorbike": "sepeda motor", "person": "orang",
    "potted plant": "tanaman pot", "sheep": "domba", "sofa": "sofa",
    "train": "kereta", "tv/monitor": "televisi/monitor",
}

def _download_if_missing(path: Path, url: str):
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".download")
        try:
            urllib.request.urlretrieve(url, temp)
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink()

class ObjectCounter:
    """CPU object detector; recognizes the 20 Pascal VOC categories."""
    def __init__(self, confidence=0.45):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV belum terpasang. Jalankan: python -m pip install opencv-python-headless") from exc
        _download_if_missing(PROTOTXT_PATH, PROTOTXT_URL)
        _download_if_missing(WEIGHTS_PATH, WEIGHTS_URL)
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(WEIGHTS_PATH))
        self.confidence = confidence

    def detect(self, image: Image.Image):
        import numpy as np
        cv2 = self.cv2
        rgb = image.convert("RGB")
        frame = np.asarray(rgb)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        output = self.net.forward()
        found = []
        for item in output[0, 0]:
            score = float(item[2])
            class_id = int(item[1])
            if score < self.confidence or not 0 < class_id < len(VOC_LABELS):
                continue
            x1, y1, x2, y2 = item[3:7]
            box = [max(0, int(x1*w)), max(0, int(y1*h)),
                   min(w-1, int(x2*w)), min(h-1, int(y2*h))]
            label = VOC_LABELS[class_id]
            found.append({"label": label, "label_id": ID_LABELS.get(label, label),
                          "confidence": score, "box": box})
        # Keep the highest-confidence result when duplicate boxes overlap heavily.
        found.sort(key=lambda x: x["confidence"], reverse=True)
        kept = []
        for candidate in found:
            x1, y1, x2, y2 = candidate["box"]
            duplicate = False
            for prior in kept:
                if candidate["label"] != prior["label"]:
                    continue
                a1, b1, a2, b2 = prior["box"]
                ix1, iy1 = max(x1, a1), max(y1, b1)
                ix2, iy2 = min(x2, a2), min(y2, b2)
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                area1 = max(1, (x2-x1)*(y2-y1))
                area2 = max(1, (a2-a1)*(b2-b1))
                if inter / max(1, area1+area2-inter) > 0.45:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def summarize(detections):
        from collections import Counter
        counts = Counter(item["label_id"] for item in detections)
        return {"total": len(detections), "counts": dict(counts)}

    def draw(self, image: Image.Image, detections):
        from PIL import ImageDraw
        output = image.convert("RGB").copy()
        draw = ImageDraw.Draw(output)
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            caption = f'{item["label_id"]} {item["confidence"]:.0%}'
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1 + 3, max(0, y1 - 16)), caption, fill="red", stroke_width=1, stroke_fill="white")
        return output
