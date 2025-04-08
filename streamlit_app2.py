import streamlit as st
import tempfile
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms import Normalize
from model.unet_generator import UNetGenerator

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_generator_model():
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load("model/best_generator_3.1.2_v4.pth", map_location=device))
    model.eval()
    return model

generator = load_generator_model()

def load_nii_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    img = nib.load(tmp_path)
    return img.get_fdata(), nib.load(tmp_path).affine  # return affine too for saving

def normalize_mri(mri_image, eps=1e-5):
    return (mri_image - np.min(mri_image)) / (np.max(mri_image) - np.min(mri_image) + eps)

def crop_or_pad_image(image, target_shape=(256, 256)):
    h, w = image.shape
    target_h, target_w = target_shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    padded = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
    return padded[:target_h, :target_w]

to_tensor = T.ToTensor()
normalizer = Normalize(mean=[0.5], std=[0.5])

st.title("🧠 MRI to CT Image Synthesis")

uploaded_file = st.file_uploader("Upload a NIfTI MRI file (.nii or .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    st.success("File uploaded successfully. Processing...")

    mri_3d, affine = load_nii_file(uploaded_file)
    mri_3d = normalize_mri(mri_3d)

    total_slices = mri_3d.shape[2]
    ct_slices = []

    # Process all slices
    with torch.no_grad():
        for idx in range(total_slices):
            mri_slice = mri_3d[:, :, idx]
            mri_slice = crop_or_pad_image(mri_slice, (256, 256))

            mri_tensor = to_tensor(mri_slice.astype(np.float32))
            mri_tensor = normalizer(mri_tensor)
            mri_tensor = mri_tensor.unsqueeze(0).to(device)

            gen_ct = generator(mri_tensor).squeeze().cpu().numpy()
            gen_ct_vis = (gen_ct + 1) / 2  # scale to [0,1]
            ct_slices.append(gen_ct)

    ct_volume = np.stack(ct_slices, axis=2)

    # Animated slice viewer
    st.subheader("🌀 Slice Viewer")
    slice_slider = st.slider("Select Slice", 0, total_slices - 1, total_slices // 2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mri_3d[:, :, slice_slider], cmap='gray')
    ax[0].set_title(f'MRI #{slice_slider}')
    ax[0].axis('off')

    ax[1].imshow((ct_volume[:, :, slice_slider] + 1) / 2, cmap='gray')
    ax[1].set_title(f'CT #{slice_slider}')
    ax[1].axis('off')

    st.pyplot(fig)

    # Save NIfTI and provide download
    ct_nifti = nib.Nifti1Image(ct_volume.astype(np.float32), affine=affine)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_out:
        nib.save(ct_nifti, tmp_out.name)
        with open(tmp_out.name, "rb") as f:
            st.download_button("📥 Download Synthesized CT NIfTI", f, file_name="synth_ct.nii.gz")
