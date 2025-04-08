# MRI to Synthetic CT Streamlit application

## How to run this Streamlit application
```
streamlit run app.py
```
## How to add new model

- Step 1 : Add new model to /model/final_generator_XXXXX.pth file
- Step 2 : Update the parameters of /model/unet_generator.py 's UNet model based on trained .pth file
- Step 3 : Update the import line in /backend/model_predict.py line 9 .
```
def load_model(model_path="model/best_generator_3.1.2_v3.pth"):
```

## File Structure

```
mri_to_ct_project/
├── backend/
│   ├── __init__.py
│   ├── nifti_2_npy.py
│   ├── preprocess.py
│   ├── model_predict.py
│   └── blob_storage.py
├── model/
│   ├── unet_generator.py
│   └── final_generator_3.1.2_r1.pth
├── frontend/
│   ├── __init__.py
│   ├── home_page.py
│   ├── result_page.py
│   └── utils.py
├── blob_storage/
├── app.py                       # Main Streamlit app
└── requirements.txt
```

## Updated file structure

```
Streamlit_MRI2CT/
├── streamlit_app(version_number).py
├── model/
│   └── unet_generator.py
│   └── best_generator_3.1.2_v4.pth
├── venv/
```

# Latest version does

## 💾  Upload Support:

Accepts NIfTI MRI files (.nii or .nii.gz) via Streamlit's file uploader.

## 🧠 MRI Preprocessing:

Normalizes the 3D MRI volume to the [0, 1] range.
Applies center-crop or zero-padding to each 2D slice to fit the model input size (256x256).

## 🧠→🦴 CT Synthesis:

Uses a pretrained UNet Generator model to convert each MRI slice into a synthetic CT slice.
Processes all slices in the volume (not just a few).

## 📊 Side-by-Side Visualization:

Shows an interactive comparison of the MRI and corresponding synthetic CT slice using a slider.
MRI and CT images are displayed side-by-side in the same matplotlib figure.

## 🎞️ Animated Slice Viewer:

A slider allows scrolling through slices of the 3D MRI/CT volume for intuitive navigation.

## 📥 NIfTI Download:

Stacks all synthetic CT slices into a 3D volume.
Saves the result as a .nii.gz NIfTI file with the original affine matrix preserved.
Provides a download button to get the full synthetic CT volume.



