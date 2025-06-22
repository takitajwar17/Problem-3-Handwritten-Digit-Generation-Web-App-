import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

# Your VAE model class (same as before)
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

# Load your trained model
print("Loading trained model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(input_dim=784, latent_dim=20).to(device)

# Load the saved weights
checkpoint = torch.load('mnist_vae_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model loaded successfully!")

# Load MNIST dataset
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# Extract digit-specific latent patterns
print("Extracting digit patterns...")
digit_latents = {i: {'mu': [], 'logvar': []} for i in range(10)}

with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        mu, logvar = model.encode(data)
        
        for digit in range(10):
            mask = labels == digit
            if mask.sum() > 0:
                digit_latents[digit]['mu'].append(mu[mask].cpu())
                digit_latents[digit]['logvar'].append(logvar[mask].cpu())
        
        # Process first 100 batches to speed up
        if batch_idx >= 100:
            break

# Calculate mean latent vectors for each digit
print("Calculating digit-specific patterns...")
digit_patterns = {}

for digit in range(10):
    if digit_latents[digit]['mu']:
        all_mu = torch.cat(digit_latents[digit]['mu'], dim=0)
        all_logvar = torch.cat(digit_latents[digit]['logvar'], dim=0)
        
        digit_patterns[digit] = {
            'mean_mu': all_mu.mean(dim=0).numpy().tolist(),
            'mean_std': torch.exp(0.5 * all_logvar).mean(dim=0).numpy().tolist(),
            'samples': len(all_mu)
        }
        
        print(f"Digit {digit}: {len(all_mu)} samples processed")

# Save digit patterns as JSON
with open('digit_patterns.json', 'w') as f:
    json.dump(digit_patterns, f, indent=2)

print("✅ Digit patterns saved to 'digit_patterns.json'")
print("You can now run your Streamlit app!")