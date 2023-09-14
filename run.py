import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split

from PIL import Image


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


class ImageTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root, "*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(self.image_files[idx])

def load_and_split_datasets(transform):
    
    dataset = datasets.ImageFolder(root=DATASET_DEVELOPMENT_DIR_PATH, transform=transform)
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    return train_dataset, val_dataset, {v: k for k, v in dataset.class_to_idx.items()}

def train_and_validate_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 5
    
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{n_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

    torch.save(model, "model.pth")

def test_model_and_generate_predictions(model, test_loader, idx_to_class_mapping):
    model.eval()
    with open('predictions.txt', 'w') as f:
        with torch.no_grad():
            for batch_idx, (data, filenames) in enumerate(test_loader):
                output = model(data)
                _, predicted = torch.max(output.data, 1)

                for i in range(len(filenames)):
                    f.write(f"{filenames[i]} {idx_to_class_mapping[predicted[i].item()]}\n")

    print(f"Prediction file generated: predictions.txt")

if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--development_dir', type=str, default=os.path.join("data", "development", "images"), help='Path to the development dataset directory')
    parser.add_argument('--testing_dir', type=str, default=os.path.join("data", "testing"), help='Path to the testing dataset directory')
    args = parser.parse_args()

    DATASET_DEVELOPMENT_DIR_PATH = args.development_dir
    DATASET_TESTING_DIR_PATH = args.testing_dir

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset, val_dataset, idx_to_class_mapping = load_and_split_datasets(transform=transform)
    
    model = SimpleCNN()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    train_and_validate_model(model, train_loader, val_loader)
    
    test_dataset = ImageTestDataset(root=DATASET_TESTING_DIR_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_model_and_generate_predictions(model, test_loader, idx_to_class_mapping)
