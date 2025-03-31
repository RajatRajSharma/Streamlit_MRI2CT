# File location: mri_to_ct_project/backend/model_predict.py

import torch
import numpy as np
from backend.blob_storage import save_to_blob
from model.unet_generator import UNetGenerator  # Make sure this import path is correct
import time

def load_model(model_path="model/best_generator_3.1.2_v3.pth"):
    """Load trained generator model for inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture
    model = UNetGenerator().to(device)
    
    # Load state_dict with proper mapping
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove 'module.' prefix if model was trained with DataParallel
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(cleaned_state_dict)
    
    # Set to evaluation mode
    model.eval()
    
    return model, device

def generate_ct_scan(preprocessed_path, output_path=None):
    """Generate CT from MRI using trained generator for all slices"""
    # Load model and device
    model, device = load_model()
    
    # Load MRI data
    mri_data = np.load(preprocessed_path)
    print(f"Input MRI shape: {mri_data.shape}")
    
    # Ensure we have 3D input (convert 2D to 3D if needed)
    if mri_data.ndim == 2:
        print("Converting 2D input to 3D (1, H, W)")
        mri_data = np.expand_dims(mri_data, axis=0)
    elif mri_data.ndim != 3:
        raise ValueError("Input must be 2D or 3D numpy array")
    
    num_slices = mri_data.shape[0]
    ct_slices = []
    
    print(f"Processing {num_slices} slices...")
    
    # Process each slice individually
    for i in range(num_slices):
        # Get current slice
        current_slice = mri_data[i]
        
        # Convert to tensor and add batch+channel dimensions
        input_tensor = torch.from_numpy(current_slice).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Process output
        output_slice = output.squeeze().cpu().numpy()  # Remove batch and channel dims
        output_slice = (output_slice + 1) / 2  # Scale from [-1,1] to [0,1]
        ct_slices.append(output_slice)
        
        # Print progress every 10 slices
        if (i+1) % 10 == 0 or (i+1) == num_slices:
            print(f"Processed slice {i+1}/{num_slices}")
    
    # Stack all slices into single 3D array
    ct_volume = np.stack(ct_slices)
    print(f"Final CT volume shape: {ct_volume.shape}")
    
    # Generate timestamp for filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save results
    output_path = save_to_blob(ct_volume, f"synthetic_ct_{timestamp}")
    print(f"Saved synthetic CT volume to: {output_path}")
    
    return output_path

# import torch
# import numpy as np
# from backend.blob_storage import save_to_blob
# from model.unet_generator import UNetGenerator  # Import your generator

# def load_model(model_path="model/final_generator_3.1.2_r1.pth"):
#     """Load trained generator model for inference"""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Change 1️⃣: Use GPU if available

#     # 1. Initialize model architecture
#     model = UNetGenerator().to(device)  # Change 2️⃣: Move model to the correct device
    
#     # 2. Load state_dict with proper mapping
#     state_dict = torch.load(model_path, map_location=device)  # Change 3️⃣: Load weights to correct device
    
#     # 3. Remove 'module.' prefix if model was trained with DataParallel
#     cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
#     # 4. Load weights
#     model.load_state_dict(cleaned_state_dict)
    
#     # 5. Set to evaluation mode
#     model.eval()
    
#     return model

# def generate_ct_scan(preprocessed_path, output_path=None):
#     """Generate CT from MRI using trained generator"""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Change 4️⃣: Ensure correct device usage
    
#     # Load model
#     model = load_model()
    
#     # Load and prepare input
#     mri_data = np.load(preprocessed_path)
    
#     print(f"Input size preprocessed_path : {mri_data.shape}")
    
#     # Handle 2D or 3D input
#     if mri_data.ndim == 2:
#         print("+++++++++ We got a 2D image")
#         # Add batch and channel dims: (1, 1, H, W)
#         input_tensor = torch.from_numpy(mri_data).float().unsqueeze(0).unsqueeze(0)
#     elif mri_data.ndim == 3:
#         print("######## We got a 3D image")
#         # Process first slice: (1, 1, H, W)
#         input_tensor = torch.from_numpy(mri_data[0]).float().unsqueeze(0).unsqueeze(0)
#     else:
#         raise ValueError("Input must be 2D or 3D numpy array")

#     print(f"Input size input_tensor : {input_tensor.shape}")
#     input_tensor = input_tensor.to(device)  # Change 5️⃣: Move input to correct device

#     # Inference
#     with torch.no_grad():
#         output = model(input_tensor)

#     print(f"Input size output : {output.shape}")

#     # Convert to numpy and scale to [0,1]
#     ct_data = output.squeeze().cpu().numpy()  # Change 6️⃣: Ensure output is moved back to CPU
#     ct_data = (ct_data + 1) / 2  # Assuming model outputs [-1,1]

#     print(f"Input size ct_data : {ct_data.shape}")

#     # Save results
#     output_path = save_to_blob(ct_data, "synthetic_ct")
    

#     return output_path
