# File location: mri_to_ct_project/backend/preprocess.py
import numpy as np
from scipy.ndimage import zoom
from backend.blob_storage import save_to_blob

def normalize_ct(ct_image: np.ndarray, clip_min: float = -1000, clip_max: float = 1000) -> np.ndarray:
    """
    Normalize CT image by clipping Hounsfield units and scaling to [0, 1]
    
    Parameters:
        ct_image: Input CT image array
        clip_min: Minimum Hounsfield unit value
        clip_max: Maximum Hounsfield unit value
        
    Returns:
        Normalized CT image
    """
    ct_image = np.clip(ct_image, clip_min, clip_max)
    return (ct_image - clip_min) / (clip_max - clip_min)

def normalize_mri(mri_image: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Normalize MRI image by scaling to [0, 1]
    
    Parameters:
        mri_image: Input MRI image array
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized MRI image
    """
    return (mri_image - np.min(mri_image)) / (np.max(mri_image) - np.min(mri_image) + eps)

def crop_or_pad_image(image: np.ndarray, target_shape: tuple = (256, 256)) -> np.ndarray:
    """
    Crop or pad image to reach target shape while maintaining content.
    Handles both 2D (height, width) and 3D (slices, height, width) arrays.
    
    Parameters:
        image: Input image array (2D or 3D)
        target_shape: Desired output shape (height, width)
        
    Returns:
        Resized image array
    """
    if image.ndim == 2:  # Single 2D image
        current_shape = image.shape
        result = np.zeros((target_shape[0], target_shape[1]), dtype=image.dtype)
    elif image.ndim == 3:  # 3D stack of images
        current_shape = image.shape[1:]  # (slices, height, width) -> (height, width)
        result = np.zeros((image.shape[0], target_shape[0], target_shape[1]), dtype=image.dtype)
    else:
        raise ValueError("Input image must be 2D or 3D")

    height_diff = current_shape[0] - target_shape[0]
    width_diff = current_shape[1] - target_shape[1]

    if image.ndim == 2:
        # Handle height dimension
        if height_diff > 0:  # Crop
            top_crop = (height_diff + 1) // 2
            bottom_crop = height_diff // 2
            cropped = image[top_crop:-bottom_crop, :]
        elif height_diff < 0:  # Pad
            top_pad = (-height_diff + 1) // 2
            bottom_pad = -height_diff // 2
            cropped = np.pad(image, ((top_pad, bottom_pad), (0, 0)), mode='constant', constant_values=0)
        else:
            cropped = image

        # Handle width dimension
        if width_diff > 0:  # Crop
            right_crop = (width_diff + 1) // 2
            left_crop = width_diff // 2
            result = cropped[:, left_crop:-right_crop]
        elif width_diff < 0:  # Pad
            right_pad = (-width_diff + 1) // 2
            left_pad = -width_diff // 2
            result = np.pad(cropped, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)
        else:
            result = cropped

    elif image.ndim == 3:
        for i in range(image.shape[0]):
            # Handle height dimension
            if height_diff > 0:  # Crop
                top_crop = (height_diff + 1) // 2
                bottom_crop = height_diff // 2
                cropped = image[i, top_crop:-bottom_crop, :]
            elif height_diff < 0:  # Pad
                top_pad = (-height_diff + 1) // 2
                bottom_pad = -height_diff // 2
                cropped = np.pad(image[i], ((top_pad, bottom_pad), (0, 0)), mode='constant', constant_values=0)
            else:
                cropped = image[i]

            # Handle width dimension
            if width_diff > 0:  # Crop
                right_crop = (width_diff + 1) // 2
                left_crop = width_diff // 2
                result[i] = cropped[:, left_crop:-right_crop]
            elif width_diff < 0:  # Pad
                right_pad = (-width_diff + 1) // 2
                left_pad = -width_diff // 2
                result[i] = np.pad(cropped, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)
            else:
                result[i] = cropped

    return result

def preprocess_npy(npy_file_path: str, output_path: str = None) -> str:
    """
    Preprocesses MRI data from .npy file with full pipeline
    
    Processing Steps:
    1. Load numpy array from file
    2. Normalize the MRI data (0-1 scaling)
    3. Crop/pad to standard size (256x256)
    4. Save processed data
    
    Parameters:
        npy_file_path: Path to input .npy file
        output_path: Optional path to save processed data
        
    Returns:
        str: Path to saved preprocessed .npy file
    """
    # Step 1: Load data
    mri_data = np.load(npy_file_path)
    
    # Step 2: Normalize
    if mri_data.ndim == 3:
        mri_data = np.stack([normalize_mri(slice_) for slice_ in mri_data])
    else:
        mri_data = normalize_mri(mri_data)
    
    # Step 3: Resize
    mri_data = crop_or_pad_image(mri_data)
    
    # Step 4: Save
    if output_path is None:
        output_path = save_to_blob(mri_data, prefix="preprocessed_mri")
    else:
        np.save(output_path, mri_data)
    
    return output_path
