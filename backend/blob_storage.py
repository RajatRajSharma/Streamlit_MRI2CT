# File location: mri_to_ct_project/backend/blob_storage.py

import os
from datetime import datetime
import numpy as np

# Local storage simulation
BLOB_STORAGE_DIR = "blob_storage"

def save_to_blob(data: np.ndarray, prefix: str = "data") -> str:
    """
    Save numpy array to blob storage with timestamp
    
    Parameters:
        data: Numpy array to save
        prefix: Filename prefix
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(BLOB_STORAGE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.npy"
    filepath = os.path.join(BLOB_STORAGE_DIR, filename)
    
    np.save(filepath, data)
    print(f"Saved file: {filepath}")
    
    return filepath