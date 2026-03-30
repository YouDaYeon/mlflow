# MLflow Deep Learning
# https://mlflow.org/docs/latest/ml/getting-started/deep-learning/

# pip install mlflow torch torchvision psutil

# Step 1: Create a new experiment
import mlflow

# The set_experiment API creates a new experiment if it doesn't exist.
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # 웹 대시보드에 기록
mlflow.set_experiment("Deep Learning Experiment")

# IMPORTANT: Enable system metrics monitoring
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)


# Step 2: Prepare the dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST("data", train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Step 3: Define the model and optimizer
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

# Training parameters
params = {
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "optimizer": "SGD",
    "model_type": "MLP",
    "hidden_units": [512, 512],
}

# Define optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])

# Step 4: Train the model
with mlflow.start_run() as run:
    # Log training parameters
    mlflow.log_params(params)

    for epoch in range(params["epochs"]):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log batch metrics (every 100 batches)
            if batch_idx % 100 == 0:
                batch_loss = train_loss / (batch_idx + 1)
                batch_acc = 100.0 * correct / total
                mlflow.log_metrics(
                    {"batch_loss": batch_loss, "batch_accuracy": batch_acc},
                    step=epoch * len(train_loader) + batch_idx,
                )

        # Calculate epoch metrics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Calculate and log epoch validation metrics
        val_loss = val_loss / len(test_loader)
        val_acc = 100.0 * val_correct / val_total

        # Log epoch metrics
        mlflow.log_metrics(
            {
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            step=epoch,
        )
        # Log checkpoint at the end of each epoch
        mlflow.pytorch.log_model(model, name=f"checkpoint_{epoch}")

        print(
            f"Epoch {epoch + 1}/{params['epochs']}, "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Log the final trained model
    model_info = mlflow.pytorch.log_model(model, name="final_model")

# ############################
# # Step 6: Load back the model and run inference
# # Load the final model
# model = mlflow.pytorch.load_model("runs:/12f40122d13a47509dffb486489230d5/final_model")
# # or load a checkpoint
# # model = mlflow.pytorch.load_model("runs:/<run_id>/checkpoint_<epoch>")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# loss_fn = nn.CrossEntropyLoss()

# # Resume the previous run to log test metrics
# run_id = "12f40122d13a47509dffb486489230d5"
# with mlflow.start_run(run_id=run_id) as run:
#     # Evaluate the model on the test set
#     test_loss, test_correct, test_total = 0, 0, 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = loss_fn(output, target)

#             test_loss += loss.item()
#             _, predicted = output.max(1)
#             test_total += target.size(0)
#             test_correct += predicted.eq(target).sum().item()

#     # Calculate and log final test metrics
#     test_loss = test_loss / len(test_loader)
#     test_acc = 100.0 * test_correct / test_total

#     mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
#     print(f"Final Test Accuracy: {test_acc:.2f}%")