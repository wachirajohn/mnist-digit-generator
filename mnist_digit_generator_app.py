import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

z_dim = 100
num_classes = 10
image_size = 28
model_path = "models/mnist_generator.pth"
device = torch.device("cpu")


# Generator Model
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


# Load model
@st.cache_resource
def load_model():
    model = Generator(z_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Generate images
def generate_images(model, digit, num_images=5):
    noise = torch.randn(num_images, z_dim).to(device)
    labels = torch.full((num_images,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        generated = model(noise, labels).cpu().numpy()

    generated = (generated + 1) / 2
    generated = np.clip(generated, 0, 1)
    return generated


# Streamlit UI
st.title("Handwritten Digit Generator (MNIST)")
digit = st.selectbox("Select a digit to generate (0–9):", list(range(10)))

if st.button("Generate 5 Samples"):
    model = load_model()
    images = generate_images(model, digit)

    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        axs[i].imshow(images[i][0], cmap="gray", vmin=0, vmax=1)
        axs[i].axis("off")
        axs[i].set_title(f"Sample {i + 1}")

    plt.suptitle(f"Generated Digit: {digit}", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)