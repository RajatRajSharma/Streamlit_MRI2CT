import nibabel as nib
import numpy as np
import os
from backend.blob_storage import save_to_blob

def convert_nifti_to_npy(nifti_file_path: str, output_path: str = None) -> str:
    """
    Converts a NIfTI file to a NumPy (.npy) file with basic processing.
    
    Steps:
    1. Load NIfTI file using nibabel
    2. Extract the image data as numpy array
    3. Save as .npy file
    
    Parameters:
        nifti_file_path (str): Path to the input NIfTI file
        output_path (str): Optional path to save the .npy file
        
    Returns:
        str: Path to the saved .npy file
    """
    # Step 1: Load NIfTI file
    img = nib.load(nifti_file_path)
    
    # Step 2: Get data as numpy array
    npy_data = img.get_fdata()
    
    # Step 3: Save to blob storage if no output path provided
    if output_path is None:
        output_path = save_to_blob(npy_data, prefix="raw_mri")
    else:
        np.save(output_path, npy_data)
    
    return output_path