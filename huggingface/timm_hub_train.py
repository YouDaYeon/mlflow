# Timm Hub 모델 학습 예제

import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import mlflow


def main():
    # MLflow experiment 설정
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Timm Hub")

    # 데이터셋 준비 (여기서는 예시로 CIFAR-10 사용)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 모델 로드 (Label 개수 및 입력 shape 맞춰야 함. CIFAR10 -> num_classes=10)
    model = timm.create_model(
        'hf-hub:nateraw/resnet50-oxford-iiit-pet',
        pretrained=True,
        num_classes=10  # CIFAR-10은 10개 클래스
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 학습 루프
    epochs = 2  # 데모용으로 에폭 수를 줄임
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # 검증 평가
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # MLflow로 로깅
        with mlflow.start_run(nested=True):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_param("lr", 1e-4)
            mlflow.log_param("optimizer", "AdamW")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("model", "hf-hub:nateraw/resnet50-oxford-iiit-pet")

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("학습 완료")


if __name__ == "__main__":
    main()
