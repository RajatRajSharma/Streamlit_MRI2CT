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
