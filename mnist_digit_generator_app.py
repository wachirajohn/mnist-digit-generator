import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

z_dim = 100
num_classes = 10
image_size = 28
model_path = "models/mnist_generator.pth"
device = torch.device("cpu")


# Generator Model - EXACTLY matching your training code
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_emb(labels)], 1)
        out = self.model(x)
        return out.view(-1, 1, image_size, image_size)


# Enhanced model loading with error handling
@st.cache_resource
def load_model():
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please upload the 'mnist_generator.pth' file to the 'models/' directory")
            return None

        # Load the model
        model = Generator(z_dim, num_classes).to(device)

        # Debug: Print model structure
        st.write("Loading model with architecture:")
        st.write(f"- Input: {z_dim + num_classes} dimensions")
        st.write(f"- Hidden layers: 256 → 512 → 1024 → {image_size * image_size}")
        st.write(f"- BatchNorm layers: Yes")

        # Load state dict with error handling
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        st.success("Model loaded successfully!")
        return model

    except RuntimeError as e:
        st.error("Model architecture mismatch!")
        st.write("**Error details:**")
        st.code(str(e))
        st.write("**Solution:** The saved model has different architecture than expected.")
        st.write("**Common causes:**")
        st.write("- BatchNorm layers missing/extra")
        st.write("- Different layer sizes")
        st.write("- Different activation functions")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# Generate digit images
def generate_images(model, digit, num_images=5):
    if model is None:
        return None

    noise = torch.randn(num_images, z_dim).to(device)
    labels = torch.full((num_images,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        generated = model(noise, labels).cpu().numpy()

    # Convert from (-1, 1) to (0, 1) range for display
    generated = (generated + 1) / 2
    generated = np.clip(generated, 0, 1)

    return generated


# Debug function to inspect saved model
def debug_model_structure():
    if st.button("Debug Model Structure"):
        try:
            state_dict = torch.load(model_path, map_location=device)
            st.write("**Saved model parameters:**")
            for key in state_dict.keys():
                st.write(f"- {key}: {state_dict[key].shape}")
        except Exception as e:
            st.error(f"Cannot load model for debugging: {e}")


# Streamlit UI
st.title("Handwritten Digit Generator (MNIST)")

digit = st.selectbox("Select a digit to generate (0–9):", list(range(10)))

if st.button("Generate 5 Samples"):
    model = load_model()

    if model is not None:
        images = generate_images(model, digit)

        if images is not None:
            # Display images
            fig, axs = plt.subplots(1, 5, figsize=(12, 3))
            for i in range(5):
                axs[i].imshow(images[i][0], cmap="gray", vmin=0, vmax=1)
                axs[i].axis("off")
                axs[i].set_title(f"Sample {i + 1}")

            plt.suptitle(f"Generated Digit: {digit}", fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Failed to generate images")
    else:
        st.error("Cannot generate images without a valid model")