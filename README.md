# 🤖 MNIST Handwritten Digit Generator

A web application that generates realistic handwritten digits using a Variational Autoencoder (VAE) trained on the MNIST dataset.

## ✨ Features

- **AI-Powered Generation**: Uses a trained VAE model to generate realistic handwritten digits
- **Digit-Specific Patterns**: Extracts and uses digit-specific latent patterns for better quality
- **Interactive Web Interface**: Clean, user-friendly Streamlit interface
- **Download Capability**: Download generated images as PNG files
- **Real-time Generation**: Generate 5 variations of any digit (0-9) instantly

## 🚀 Live Demo

[View the deployed app here](https://your-app-url.streamlit.app) _(Replace with actual deployment URL)_

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── extract_patterns.py       # Script to extract digit-specific patterns
├── mnist_vae_model.pth       # Pre-trained VAE model weights
├── digit_patterns.json       # Extracted digit patterns (generated)
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml    # Streamlit configuration
└── README.md                 # This file
```

## 🛠️ Local Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Problem-3-Handwritten-Digit-Generation-Web-App-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Extract digit patterns** (if not already done)
   ```bash
   python extract_patterns.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Option 2: Railway
1. Push to GitHub
2. Visit [railway.app](https://railway.app)
3. Connect repository and deploy

### Option 3: Render
1. Push to GitHub
2. Visit [render.com](https://render.com)
3. Create new web service from repository

### Option 4: Heroku
1. Install Heroku CLI
2. Create `Procfile`: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
3. Deploy to Heroku

## 🧠 Model Architecture

The application uses a Variational Autoencoder (VAE) with:
- **Encoder**: 784 → 400 → 20 (latent dimensions)
- **Decoder**: 20 → 400 → 784
- **Latent Space**: 20-dimensional for efficient representation
- **Training**: MNIST dataset (60,000 training images)

## 📊 How It Works

1. **Pattern Extraction**: `extract_patterns.py` analyzes the trained model and extracts digit-specific latent patterns
2. **Generation Process**: For each digit, the app samples from the learned latent distribution
3. **Decoding**: The VAE decoder converts latent vectors back to 28×28 pixel images
4. **Display**: Generated images are displayed in the web interface

## 🔧 Configuration

The app includes several configurations in `.streamlit/config.toml`:
- Optimized for web deployment
- Custom theme colors
- Performance settings

## 📝 Dependencies

- `streamlit>=1.28.0` - Web framework
- `torch>=2.0.0` - PyTorch for model inference
- `torchvision>=0.15.0` - Computer vision utilities
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Plotting and visualization
- `Pillow>=9.0.0` - Image processing

## 🐛 Common Issues

### OpenMP Runtime Error
If you encounter OpenMP initialization errors:
- The app automatically sets `KMP_DUPLICATE_LIB_OK=TRUE`
- This is a known issue with PyTorch on Windows

### Model File Size
- The model file (`mnist_vae_model.pth`) is ~1MB
- Most deployment platforms support this size
- For GitHub, ensure Git LFS is configured if needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Built with ❤️ using PyTorch and Streamlit

---

**Note**: Make sure to run `extract_patterns.py` before deploying to generate the `digit_patterns.json` file required for optimal digit generation. 