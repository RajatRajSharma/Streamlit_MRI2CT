# MRI to Synthetic CT Streamlit application

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
