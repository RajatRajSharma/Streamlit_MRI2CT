import streamlit as st
import tempfile
import numpy as np
import io
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms import Normalize
from model.unet_generator import UNetGenerator  # <- Your model class

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained Generator
@st.cache_resource
def load_generator_model():
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load("model/best_generator_3.1.2_v4.pth", map_location=device))
    model.eval()
    return model

generator = load_generator_model()

def load_nii_file(uploaded_file):
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load using nibabel
    img = nib.load(tmp_path)
    return img.get_fdata()

def normalize_mri(mri_image, eps=1e-5):
    return (mri_image - np.min(mri_image)) / (np.max(mri_image) - np.min(mri_image) + eps)

def crop_or_pad_image(image, target_shape=(256, 256)):
    h, w = image.shape
    target_h, target_w = target_shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    padded = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
    cropped = padded[:target_h, :target_w]
    return cropped

# Define the normalization transform (to match training)
to_tensor = T.ToTensor()  # Converts HxW numpy to tensor [1, H, W] and scales to [0, 1]
normalizer = Normalize(mean=[0.5], std=[0.5])  # Scales to [-1, 1]

# === Streamlit UI ===
st.title("🧠 MRI to CT Image Synthesis")

uploaded_file = st.file_uploader("Upload a NIfTI MRI file (.nii or .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    st.success("File uploaded successfully. Processing...")

    # Step 1: Read NIfTI MRI file
    mri_3d = load_nii_file(uploaded_file)
    mri_3d = normalize_mri(mri_3d)  # Normalize to [0, 1]

    # Step 2: Choose central slices (up to 4)
    total_slices = mri_3d.shape[2]
    slice_indices = np.linspace(total_slices // 3, 2 * total_slices // 3, 4, dtype=int)

    # Two Streamlit columns for display
    col1, col2 = st.columns(2)

    with torch.no_grad():
        for i, idx in enumerate(slice_indices):
            mri_slice = mri_3d[:, :, idx]
            mri_slice = crop_or_pad_image(mri_slice, (256, 256))

            # Convert to tensor and normalize to [-1, 1]
            mri_tensor = to_tensor(mri_slice.astype(np.float32))  # [1, H, W], range [0, 1]
            mri_tensor = normalizer(mri_tensor)  # Normalize to [-1, 1]
            mri_tensor = mri_tensor.unsqueeze(0).to(device)  # [1, 1, H, W]

            # Run Generator
            gen_ct = generator(mri_tensor).squeeze().cpu().numpy()

            # Rescale for visualization
            mri_vis = (mri_tensor.squeeze().cpu().numpy() + 1) / 2  # [-1,1] → [0,1]
            gen_ct_vis = (gen_ct + 1) / 2

            # Plot MRI and CT side-by-side
            fig_mri, ax_mri = plt.subplots()
            ax_mri.imshow(mri_vis, cmap='gray')
            ax_mri.axis('off')
            ax_mri.set_title(f"MRI Slice #{idx}")

            fig_ct, ax_ct = plt.subplots()
            ax_ct.imshow(gen_ct_vis, cmap='gray')
            ax_ct.axis('off')
            ax_ct.set_title(f"Synth CT #{idx}")

            # Alternate display in columns
            if i % 2 == 0:
                col1.pyplot(fig_mri)
                col1.pyplot(fig_ct)
            else:
                col2.pyplot(fig_mri)
                col2.pyplot(fig_ct)
