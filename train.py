import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3 pada dataset sendiri.")
    parser.add_argument("--data_dir", default="data/train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="models/custom_mobilenetv3.pth")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs dan batch_size harus minimal 1")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    if len(ds.classes) < 2 or len(ds) < 2:
        raise ValueError("Dataset memerlukan minimal dua kelas dan dua gambar.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(ds.classes))
    model.cpu()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total = correct = seen = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x.cpu())
            loss = loss_fn(out, y.cpu())
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        print(f"Epoch {epoch+1}/{args.epochs} loss={total/seen:.4f} train_acc={correct/seen:.1%}")
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": ds.classes}, dest)
    print(f"Model tersimpan: {dest.resolve()}")

if __name__ == "__main__":
    main()
