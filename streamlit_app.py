# import streamlit as st
# import numpy as np
# import torch
# import random
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as T
# from torchvision.transforms import Normalize

# # === Load your Generator model architecture ===
# from model.unet_generator import UNetGenerator  # <- replace with your model class

# # === Device ===
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # === Load Pretrained Generator ===
# @st.cache_resource
# def load_generator_model():
#     model = UNetGenerator().to(device)
#     model.load_state_dict(torch.load("model/best_generator_3.1.2_v4.pth", map_location=device))
#     model.eval()
#     return model

# generator = load_generator_model()

# # === Load MRI input data ===
# @st.cache_resource
# def load_test_input():
#     input_data = np.load('./mini data/data(brain-new-36)/Test_Patients_InputData/1BA091_input.npy').astype(np.float32)
#     return input_data

# # === Transformations ===
# class PairedTransform:
#     def __init__(self):
#         self.transforms = T.Compose([
#             T.RandomHorizontalFlip(),
#             T.RandomVerticalFlip(),
#             T.RandomRotation(degrees=10)
#         ])
#         self.normalize = Normalize(mean=[0.5], std=[0.5])

#     def __call__(self, mri_image):
#         seed = random.randint(0, 2**32)
#         torch.manual_seed(seed)
#         mri_image = self.transforms(mri_image)
#         return self.normalize(mri_image)

# # === Dataset Class ===
# class MRIDataset(Dataset):
#     def __init__(self, input_data, transform=None):
#         self.input_data = input_data
#         self.transform = transform

#     def __len__(self):
#         return len(self.input_data)

#     def __getitem__(self, idx):
#         mri_image = torch.tensor(self.input_data[idx][np.newaxis, :, :])  # shape: [1, H, W]
#         if self.transform:
#             mri_image = self.transform(mri_image)
#         return mri_image

# # === UI Header ===
# st.title("🧠 MRI to CT Image Synthesis")

# input_data = load_test_input()
# transform = PairedTransform()
# mri_dataset = MRIDataset(input_data, transform=transform)
# mri_loader = DataLoader(mri_dataset, batch_size=1, shuffle=False)

# # === Visualize MRI and Synthesized CT ===
# if st.button("🎨 Generate and Display Synthetic CT"):
#     fig, axes = plt.subplots(2, 4, figsize=(16, 8))

#     with torch.no_grad():
#         for idx in range(4):
#             mri_image = mri_dataset[idx]
#             mri_input = mri_image.unsqueeze(0).to(device)
#             gen_ct = generator(mri_input).squeeze(0)

#             # Rescale for visualization
#             mri_image_vis = (mri_input.squeeze(0).cpu() + 1) / 2
#             gen_ct_vis = (gen_ct.cpu() + 1) / 2

#             axes[0, idx].imshow(mri_image_vis.squeeze(), cmap='gray')
#             axes[0, idx].axis('off')
#             axes[0, idx].set_title(f"Real MRI #{idx + 1}")

#             axes[1, idx].imshow(gen_ct_vis.squeeze(), cmap='gray')
#             axes[1, idx].axis('off')
#             axes[1, idx].set_title(f"Synth CT #{idx + 1}")

#     st.pyplot(fig)




import streamlit as st
import tempfile
import numpy as np
import io
import torch
import nibabel as nib
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
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

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

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

            axes[0, i].imshow(mri_vis, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Real MRI #{idx}")

            axes[1, i].imshow(gen_ct_vis, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Synth CT #{idx}")

    st.pyplot(fig)
