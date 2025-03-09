import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from cnn_model import CNNModel

def train_model():
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Load CIFAR-10 dataset and take only the first 300 samples
    full_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataset = Subset(full_dataset, range(300))  # Take first 300 images
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Load model
    model = CNNModel()
    model.train()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 3   # Fewer epochs for quick training
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Save trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to 'model.pth'")

if __name__ == "__main__":
    train_model()

