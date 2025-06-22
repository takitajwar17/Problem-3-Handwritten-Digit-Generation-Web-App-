import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 50
latent_dim = 20
input_dim = 28 * 28

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# VAE Model Architecture
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function (ELBO)
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize model
model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
train_losses = []

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('VAE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Function to generate digits
def generate_digit_samples(model, digit, num_samples=5, device=device):
    """Generate samples for a specific digit"""
    model.eval()
    
    # Get samples of the specific digit from training data
    digit_samples = []
    with torch.no_grad():
        for data, labels in train_loader:
            mask = labels == digit
            if mask.sum() > 0:
                digit_data = data[mask][:min(100, mask.sum())]
                digit_samples.append(digit_data)
            if len(digit_samples) >= 10:  # Collect enough samples
                break
    
    if not digit_samples:
        print(f"No samples found for digit {digit}")
        return None
    
    # Concatenate all digit samples
    digit_samples = torch.cat(digit_samples, dim=0).to(device)
    
    # Encode to get the latent space distribution for this digit
    mu, logvar = model.encode(digit_samples)
    
    # Calculate mean and std of the latent distribution for this digit
    digit_mu = mu.mean(dim=0)
    digit_std = torch.exp(0.5 * logvar).mean(dim=0)
    
    # Generate new samples by sampling from the digit's latent distribution
    generated_samples = []
    for _ in range(num_samples):
        # Sample from the digit's latent distribution
        z = torch.randn(1, model.fc21.out_features).to(device)
        z = digit_mu + z * digit_std
        
        # Decode to generate image
        with torch.no_grad():
            sample = model.decode(z)
            generated_samples.append(sample.cpu().numpy().reshape(28, 28))
    
    return generated_samples

# Test generation for all digits
print("Testing digit generation...")
for digit in range(10):
    samples = generate_digit_samples(model, digit, num_samples=1)
    if samples:
        plt.figure(figsize=(2, 2))
        plt.imshow(samples[0], cmap='gray')
        plt.title(f'Generated Digit {digit}')
        plt.axis('off')
        plt.show()

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'latent_dim': latent_dim,
    'input_dim': input_dim,
    'model_architecture': 'VAE'
}, 'mnist_vae_model.pth')

print("Model saved as 'mnist_vae_model.pth'")

# Download the model file
files.download('mnist_vae_model.pth')

print("Training completed! Model downloaded to your local machine.")