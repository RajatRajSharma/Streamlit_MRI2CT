import streamlit as st
import numpy as np
from PIL import Image
import os
from backend.blob_storage import BLOB_STORAGE_DIR
from backend.model_predict import generate_ct_scan

def show_result_page():
    st.title("MRI to CT Synthesis - Results")

    if "preprocessed_path" not in st.session_state:
        st.warning("Please upload and preprocess a file on the Home page first.")
        return

    preprocessed_path = st.session_state["preprocessed_path"]

    # Generate synthetic CT scan and get saved path
    synthetic_ct_path = generate_ct_scan(preprocessed_path)

    if not os.path.exists(synthetic_ct_path):
        st.error("Failed to generate synthetic CT scan.")
        return
    
    # Retrieve the saved synthetic CT file from storage
    synthetic_ct_filename = os.path.basename(synthetic_ct_path)
    synthetic_ct_path = os.path.join(BLOB_STORAGE_DIR, synthetic_ct_filename)

    if not os.path.exists(synthetic_ct_path):
        st.error("Synthetic CT scan file not found in storage.")
        return

    # Load preprocessed MRI and synthetic CT
    mri_data = np.load(preprocessed_path)
    ct_data = np.load(synthetic_ct_path)

    # Handle dimensionality (2D or 3D arrays)
    num_slices = mri_data.shape[0] if mri_data.ndim == 3 else 1

    # Select 6 equally distributed indices
    if num_slices >= 6:
        step = num_slices // 6  
        indices = [i * step + step // 2 for i in range(6)]
        indices = [min(max(0, idx), num_slices - 1) for idx in indices]
    elif num_slices > 1:
        indices = np.linspace(0, num_slices - 1, min(6, num_slices)).astype(int).tolist()
    else:
        indices = [0]

    st.subheader("MRI and Synthetic CT Scans (Side-by-Side)")

    # Print dimensions
    st.write(f"MRI Shape: {mri_data.shape}")
    st.write(f"Synthetic CT Shape: {ct_data.shape}")
    st.write(f"Total MRI Slices: {num_slices}, Total CT Slices: {ct_data.shape[0] if ct_data.ndim == 3 else 1}")

    cols = st.columns(2)  # Two columns for side-by-side display
    for i, idx in enumerate(indices[:6]):  
        mri_img = Image.fromarray((mri_data[idx] * 255).astype(np.uint8)) if num_slices > 1 else Image.fromarray((mri_data * 255).astype(np.uint8))
        ct_img = Image.fromarray((ct_data[idx] * 255).astype(np.uint8)) if num_slices > 1 else Image.fromarray((ct_data * 255).astype(np.uint8))

        with cols[0]:
            st.image(mri_img, caption=f"MRI Slice {idx}", use_container_width=True)
        with cols[1]:
            st.image(ct_img, caption=f"Synthetic CT Slice {idx}", use_container_width=True)


# import streamlit as st
# import numpy as np
# from PIL import Image
# from backend.blob_storage import BLOB_STORAGE_DIR
# from backend.model_predict import generate_ct_scan
# import os

# def show_result_page():
#     st.title("MRI to CT Synthesis - Results")

#     if "preprocessed_path" not in st.session_state:
#         st.warning("Please upload and preprocess a file on the Home page first.")
#         return

#     preprocessed_path = st.session_state["preprocessed_path"]

#     # Generate synthetic CT scan if not already available
#     synthetic_ct_path = generate_ct_scan(preprocessed_path)
#     st.session_state["synthetic_ct_path"] = synthetic_ct_path

#     if not os.path.exists(synthetic_ct_path):
#         st.error("Failed to generate synthetic CT scan.")
#         return
    
    
#     synthetic_ct_path = os.path.join(BLOB_STORAGE_DIR, synthetic_ct_files[0])

#     # Load preprocessed MRI and synthetic CT
#     mri_data = np.load(preprocessed_path)
#     ct_data = np.load(synthetic_ct_path)
    
#     # Handle dimensionality (2D or 3D arrays)
#     num_slices = mri_data.shape[0] if mri_data.ndim == 3 else 1
    
#     # Select 6 equally distributed indices
#     if num_slices >= 6:
#         step = num_slices // 6  
#         indices = [i * step + step // 2 for i in range(6)]
#         indices = [min(max(0, idx), num_slices - 1) for idx in indices]
#     elif num_slices > 1:
#         indices = np.linspace(0, num_slices - 1, min(6, num_slices)).astype(int).tolist()
#     else:
#         indices = [0]

#     st.subheader("MRI and Synthetic CT Scans (Side-by-Side)")
    
#     # Print dimensions
#     st.write(f"MRI Shape: {mri_data.shape}")
#     st.write(f"Synthetic CT Shape: {ct_data.shape}")
#     st.write(f"Total MRI Slices: {num_slices}, Total CT Slices: {ct_data.shape[0] if ct_data.ndim == 3 else 1}")

#     cols = st.columns(2)  # Two columns for side-by-side display
#     for i, idx in enumerate(indices[:6]):  
#         mri_img = Image.fromarray((mri_data[idx] * 255).astype(np.uint8)) if num_slices > 1 else Image.fromarray((mri_data * 255).astype(np.uint8))
#         ct_img = Image.fromarray((ct_data[idx] * 255).astype(np.uint8)) if num_slices > 1 else Image.fromarray((ct_data * 255).astype(np.uint8))

#         with cols[0]:
#             st.image(mri_img, caption=f"MRI Slice {idx}", use_column_width=True)
#         with cols[1]:
#             st.image(ct_img, caption=f"Synthetic CT Slice {idx}", use_column_width=True)

