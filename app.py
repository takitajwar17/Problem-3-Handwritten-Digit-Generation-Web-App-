import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import json

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🤖",
    layout="centered"
)

# Your VAE model class (same as training script)
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

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VAE(input_dim=784, latent_dim=20).to(device)
        
        # Load the saved weights
        checkpoint = torch.load('mnist_vae_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_digit_patterns():
    """Load digit-specific latent patterns"""
    try:
        with open('digit_patterns.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("digit_patterns.json not found. Using random generation.")
        return None

def generate_digit_images(model, device, digit, digit_patterns=None, num_samples=5):
    """Generate images for a specific digit"""
    generated_images = []
    
    with torch.no_grad():
        for i in range(num_samples):
            if digit_patterns and str(digit) in digit_patterns:
                # Use digit-specific patterns
                mean_mu = torch.tensor(digit_patterns[str(digit)]['mean_mu']).to(device)
                mean_std = torch.tensor(digit_patterns[str(digit)]['mean_std']).to(device)
                
                # Sample from the digit's latent distribution
                z = torch.randn(1, 20).to(device)
                z = mean_mu.unsqueeze(0) + z * mean_std.unsqueeze(0) * 0.8
            else:
                # Fallback: random latent vector
                z = torch.randn(1, 20).to(device)
            
            # Decode to generate image
            generated = model.decode(z)
            
            # Convert to numpy array
            img_array = generated.cpu().numpy().reshape(28, 28)
            generated_images.append(img_array)
    
    return generated_images

def main():
    st.title("🤖 MNIST Digit Generator")
    st.markdown("Generate handwritten digits using a Variational Autoencoder (VAE)")
    
    # Info box
    with st.expander("ℹ️ How it works"):
        st.write("""
        This app uses a Variational Autoencoder (VAE) trained on the MNIST dataset to generate 
        realistic handwritten digits. The model learned to encode digits into a latent space 
        and can generate new variations by sampling from this learned distribution.
        """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, device = load_model()
        digit_patterns = load_digit_patterns()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'mnist_vae_model.pth' is in the same directory.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Digit selection
    st.markdown("### Select a digit to generate:")
    digit = st.selectbox("Choose digit (0-9):", options=list(range(10)), index=0)
    
    # Generate button
    if st.button("🎨 Generate 5 Images", type="primary"):
        with st.spinner("Generating images..."):
            try:
                images = generate_digit_images(model, device, digit, digit_patterns, 5)
                
                st.markdown(f"### Generated Images for Digit {digit}:")
                
                # Display images in a row
                cols = st.columns(5)
                for i, img in enumerate(images):
                    with cols[i]:
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(img, cmap='gray')
                        ax.axis('off')
                        ax.set_title(f'Sample {i+1}', fontsize=10)
                        st.pyplot(fig)
                        plt.close()
                
                # Download option
                st.markdown("### Download Images")
                for i, img in enumerate(images):
                    # Convert to PIL Image
                    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                    
                    # Save to bytes
                    buf = io.BytesIO()
                    img_pil.save(buf, format='PNG')
                    
                    st.download_button(
                        label=f"Download Sample {i+1}",
                        data=buf.getvalue(),
                        file_name=f"generated_digit_{digit}_sample_{i+1}.png",
                        mime="image/png"
                    )
                
            except Exception as e:
                st.error(f"Error generating images: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with PyTorch VAE and Streamlit")

if __name__ == "__main__":
    main()