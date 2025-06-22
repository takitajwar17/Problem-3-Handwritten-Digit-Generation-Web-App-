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
    """
    Variational Autoencoder for MNIST digit generation
    Architecture: 784 -> 400 -> 20 (latent) -> 400 -> 784
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar layer
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """Encode input to latent space parameters"""
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image"""
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

@st.cache_resource
def load_model():
    """Load and cache the trained VAE model"""
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
    """Load digit-specific latent patterns for better generation"""
    try:
        with open('digit_patterns.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("digit_patterns.json not found. Using random generation.")
        return None

def generate_digit_images(model, device, digit, digit_patterns=None, num_samples=5):
    """
    Generate images for a specific digit using VAE
    
    Args:
        model: Trained VAE model
        device: PyTorch device
        digit: Target digit (0-9)
        digit_patterns: Digit-specific latent patterns
        num_samples: Number of images to generate
    
    Returns:
        List of generated image arrays
    """
    generated_images = []
    
    with torch.no_grad():
        for i in range(num_samples):
            if digit_patterns and str(digit) in digit_patterns:
                # Use digit-specific patterns for better quality
                mean_mu = torch.tensor(digit_patterns[str(digit)]['mean_mu']).to(device)
                mean_std = torch.tensor(digit_patterns[str(digit)]['mean_std']).to(device)
                
                # Sample from the digit's latent distribution
                z = torch.randn(1, 20).to(device)
                z = mean_mu.unsqueeze(0) + z * mean_std.unsqueeze(0) * 0.8
            else:
                # Fallback: random latent vector
                z = torch.randn(1, 20).to(device)
            
            # Decode latent vector to generate image
            generated = model.decode(z)
            
            # Convert to numpy array and reshape to 28x28
            img_array = generated.cpu().numpy().reshape(28, 28)
            generated_images.append(img_array)
    
    return generated_images

def main():
    """Main Streamlit application"""
    st.title("🤖 MNIST Digit Generator")
    st.markdown("Generate handwritten digits using a Variational Autoencoder (VAE)")
    
    # Information about the app
    with st.expander("ℹ️ How it works"):
        st.write("""
        This app uses a Variational Autoencoder (VAE) trained on the MNIST dataset to generate 
        realistic handwritten digits. The model learned to encode digits into a latent space 
        and can generate new variations by sampling from this learned distribution.
        
        **Architecture:**
        - Encoder: 784 → 400 → 20 (latent dimensions)
        - Decoder: 20 → 400 → 784
        - Training: 60,000 MNIST images
        
        **Features:**
        - Digit-specific pattern extraction for better quality
        - Real-time generation
        - Download capability
        """)
    
    # Load model and patterns
    with st.spinner("Loading AI model..."):
        model, device = load_model()
        digit_patterns = load_digit_patterns()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'mnist_vae_model.pth' is in the same directory.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Display model info
    if digit_patterns:
        st.info("🎯 Using digit-specific patterns for enhanced generation quality!")
    else:
        st.info("⚡ Using random generation (run extract_patterns.py for better quality)")
    
    # User interface
    st.markdown("### Select a digit to generate:")
    
    # Digit selection with better UI
    col1, col2 = st.columns([1, 3])
    
    with col1:
        digit = st.selectbox(
            "Choose digit:", 
            options=list(range(10)), 
            index=0,
            help="Select which digit (0-9) you want to generate"
        )
    
    with col2:
        st.markdown(f"**Selected: {digit}**")
        if digit_patterns and str(digit) in digit_patterns:
            samples_count = digit_patterns[str(digit)].get('samples', 0)
            st.caption(f"Pattern trained on {samples_count:,} samples")
    
    # Generation controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        generate_btn = st.button("🎨 Generate 5 Images", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Regenerate", help="Generate new variations"):
            generate_btn = True
    
    # Generate and display images
    if generate_btn:
        with st.spinner("Generating images..."):
            try:
                images = generate_digit_images(model, device, digit, digit_patterns, 5)
                
                st.markdown(f"### Generated Images for Digit **{digit}**:")
                
                # Display images in a responsive grid
                cols = st.columns(5)
                for i, img in enumerate(images):
                    with cols[i]:
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(img, cmap='gray', interpolation='nearest')
                        ax.axis('off')
                        ax.set_title(f'Sample {i+1}', fontsize=10, pad=5)
                        
                        # Add border for better visibility
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(0.5)
                            spine.set_edgecolor('gray')
                        
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                
                # Download section
                st.markdown("### 📥 Download Images")
                
                # Create download buttons in columns
                download_cols = st.columns(5)
                for i, img in enumerate(images):
                    with download_cols[i]:
                        # Convert to PIL Image for download
                        img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                        
                        # Save to bytes buffer
                        buf = io.BytesIO()
                        img_pil.save(buf, format='PNG')
                        
                        st.download_button(
                            label=f"📎 Sample {i+1}",
                            data=buf.getvalue(),
                            file_name=f"generated_digit_{digit}_sample_{i+1}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                # Statistics
                st.markdown("### 📊 Generation Stats")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Generated Images", "5")
                
                with col2:
                    st.metric("Image Resolution", "28×28")
                
                with col3:
                    model_type = "Pattern-based" if digit_patterns else "Random"
                    st.metric("Generation Type", model_type)
                
            except Exception as e:
                st.error(f"Error generating images: {e}")
                st.exception(e)
    
    # Footer with additional info
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🧠 Model Info**")
        st.caption("VAE with 20D latent space")
    
    with col2:
        st.markdown("**📊 Dataset**")
        st.caption("MNIST handwritten digits")
    
    with col3:
        st.markdown("**🔧 Framework**")
        st.caption("PyTorch + Streamlit")
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p>Built with ❤️ using PyTorch and Streamlit</p>
        <p><small>Variational Autoencoder for MNIST Digit Generation</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 