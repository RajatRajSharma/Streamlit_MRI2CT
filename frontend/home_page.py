import streamlit as st
import os
import numpy as np
from PIL import Image
from backend.nifti_2_npy import convert_nifti_to_npy
from backend.preprocess import preprocess_npy

def show_home_page():
    st.title("MRI to CT Synthesis - Home")
    st.write("Upload a NIfTI file to convert it to a synthetic CT scan.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a NIfTI file", type=["nii", "nii.gz"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Step 1: Convert NIfTI to NPY
        npy_path = convert_nifti_to_npy(temp_path)
        os.remove(temp_path)  # Clean up temp file
        
        # Step 2: Preprocess the NPY file
        preprocessed_path = preprocess_npy(npy_path)
        
        # Step 3: Load preprocessed data and display 4 random images
        preprocessed_data = np.load(preprocessed_path)
        num_slices = preprocessed_data.shape[0] if preprocessed_data.ndim == 3 else 1
        
        # Print number of scans and image size
        if num_slices > 1:
            slice_shape = preprocessed_data.shape[1:]  # (height, width)
            st.write(f"Number of scans in NIfTI file: {num_slices}")
            st.write(f"Size of each scan: {slice_shape[0]} x {slice_shape[1]} pixels")
        else:
            slice_shape = preprocessed_data.shape  # (height, width)
            st.write("Number of scans in NIfTI file: 1")
            st.write(f"Size of each scan: {slice_shape[0]} x {slice_shape[1]} pixels")
        
        # Select 4 random indices with minimum distance
        if num_slices > 4:
            min_distance = num_slices // 5
            random_indices = []
            attempts = 0
            max_attempts = 100
            
            while len(random_indices) < 4 and attempts < max_attempts:
                candidate = np.random.randint(0, num_slices)
                if all(abs(candidate - idx) >= min_distance for idx in random_indices):
                    random_indices.append(candidate)
                attempts += 1
            
            if len(random_indices) < 4:
                random_indices = np.random.choice(num_slices, 4, replace=False).tolist()
        elif num_slices > 1:
            random_indices = np.random.choice(num_slices, min(4, num_slices), replace=False).tolist()
        else:
            random_indices = [0]
        
        # Display 4 random preprocessed images
        st.subheader("Preprocessed MRI Slices")
        cols = st.columns(2)
        for i, idx in enumerate(random_indices[:4]):
            if num_slices > 1:
                img = preprocessed_data[idx]
            else:
                img = preprocessed_data
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            col_idx = i % 2
            cols[col_idx].image(img, caption=f"Slice {idx if num_slices > 1 else 'Single'}", use_column_width=True)
        
        # Save preprocessed path in session state
        st.session_state["preprocessed_path"] = preprocessed_path
        st.success("Preprocessing complete!")
        
        # Synthesize CT button
        if st.button("Synthesize CT and Show Results"):
            st.session_state["navigate_to_results"] = True
            st.rerun()  # Trigger rerun to switch to Results page


# import streamlit as st
# import os
# import numpy as np
# from PIL import Image
# from backend.nifti_2_npy import convert_nifti_to_npy
# from backend.preprocess import preprocess_npy

# def show_home_page():
#     st.title("MRI to CT Synthesis - Home")
#     st.write("Upload a NIfTI file to convert it to a synthetic CT scan.")
    
#     # File upload
#     uploaded_file = st.file_uploader("Choose a NIfTI file", type=["nii", "nii.gz"])
    
#     if uploaded_file is not None:
#         # Save uploaded file temporarily
#         temp_path = f"temp_{uploaded_file.name}"
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # Step 1: Convert NIfTI to NPY
#         npy_path = convert_nifti_to_npy(temp_path)
#         os.remove(temp_path)  # Clean up temp file
        
#         # Step 2: Preprocess the NPY file
#         preprocessed_path = preprocess_npy(npy_path)
        
#         # Step 3: Load preprocessed data and display 4 random images
#         preprocessed_data = np.load(preprocessed_path)
#         num_slices = preprocessed_data.shape[0] if preprocessed_data.ndim == 3 else 1
        
#         # Print number of scans and image size
#         if num_slices > 1:
#             slice_shape = preprocessed_data.shape[1:]  # (height, width)
#             st.write(f"Number of scans in NIfTI file: {num_slices}")
#             st.write(f"Size of each scan: {slice_shape[0]} x {slice_shape[1]} pixels")
#         else:
#             slice_shape = preprocessed_data.shape  # (height, width)
#             st.write("Number of scans in NIfTI file: 1")
#             st.write(f"Size of each scan: {slice_shape[0]} x {slice_shape[1]} pixels")
        
#         # Select 4 random indices with minimum distance
#         if num_slices > 4:
#             min_distance = num_slices // 5  # Ensure some spread, adjust as needed
#             random_indices = []
#             attempts = 0
#             max_attempts = 100  # Prevent infinite loop
            
#             while len(random_indices) < 4 and attempts < max_attempts:
#                 candidate = np.random.randint(0, num_slices)
#                 if all(abs(candidate - idx) >= min_distance for idx in random_indices):
#                     random_indices.append(candidate)
#                 attempts += 1
            
#             # If we couldn't find 4 with min distance, fall back to random choice
#             if len(random_indices) < 4:
#                 random_indices = np.random.choice(num_slices, 4, replace=False).tolist()
#         elif num_slices > 1:
#             random_indices = np.random.choice(num_slices, min(4, num_slices), replace=False).tolist()
#         else:
#             random_indices = [0]  # Single slice case
        
#         # Display 4 random preprocessed images
#         st.subheader("Preprocessed MRI Slices")
#         cols = st.columns(2)  # 2 columns, 2 images per column
#         for i, idx in enumerate(random_indices[:4]):  # Limit to 4 images
#             if num_slices > 1:
#                 img = preprocessed_data[idx]
#             else:
#                 img = preprocessed_data  # 2D case
#             img = (img * 255).astype(np.uint8)  # Scale to 0-255 (assuming normalized 0-1)
#             img = Image.fromarray(img)
#             col_idx = i % 2  # Alternate between columns
#             cols[col_idx].image(img, caption=f"Slice {idx if num_slices > 1 else 'Single'}", use_column_width=True)
        
#         # Save preprocessed path in session state for result page
#         st.session_state["preprocessed_path"] = preprocessed_path
#         st.success("Preprocessing complete! Navigate to the Results page.")