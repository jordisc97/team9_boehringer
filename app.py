import gradio as gr
import pandas as pd
from fastai.vision.all import load_learner
import lightgbm as lgb

# Load pre-trained models
resnet34_model = load_learner('./models/resnet34_model.pkl')
squeezenet_model = load_learner('./models/squeezenet1_0_model.pkl')
densenet_model = load_learner('./models/densenet121_model.pkl')
lgb_model = lgb.Booster(model_file='./models/lightgbm_model.txt')

# Load the patient data
data = pd.read_csv('./data/sample_patient_data.csv')

def predict_fibrosis(age, fvc, sex, smoking_status, min_fvc, patient_id):
    """Predict pulmonary fibrosis probability based on input features.

    Args:
        age (float): Age of the patient.
        fvc (float): Forced Vital Capacity.
        sex (str): Male or Female.
        smoking_status (str): Smoking status.
        min_fvc (float): Minimum FVC of the patient.
        patient_id (str): Patient ID to fetch the associated image.

    Returns:
        dict: Predicted probabilities from the models and final probability.
    """
    # Retrieve the image path for the patient
    image_path = data.loc[data['Patient'] == patient_id, 'ImagePath'].values[0]

    # Image-based predictions
    resnet_pred, _, resnet_probs = resnet34_model.predict(image_path)
    squeezenet_pred, _, squeezenet_probs = squeezenet_model.predict(image_path)
    densenet_pred, _, densenet_probs = densenet_model.predict(image_path)

    # Aggregate image probabilities
    aggregated_image_probability = (resnet_probs[1] + squeezenet_probs[1] + densenet_probs[1]) / 3

    # Metadata features for LightGBM
    metadata_features = {
        "Age": age,
        "FVC": fvc,
        "Sex_Male": 1 if sex == "Male" else 0,
        "SmokingStatus_Ex-smoker": 1 if smoking_status == "Ex-smoker" else 0,
        "SmokingStatus_Never smoked": 1 if smoking_status == "Never smoked" else 0,
        "SmokingStatus_Currently smokes": 1 if smoking_status == "Currently smokes" else 0,
        "min_FVC": min_fvc,
        "Image_Prediction": aggregated_image_probability,
    }

    # Predict using LightGBM
    metadata_df = pd.DataFrame([metadata_features])
    final_probability = lgb_model.predict(metadata_df)[0]

    return {
        "ResNet34 Probability": round(resnet_probs[1].item(), 4),
        "SqueezeNet Probability": round(squeezenet_probs[1].item(), 4),
        "DenseNet Probability": round(densenet_probs[1].item(), 4),
        "Aggregated Image Probability": round(aggregated_image_probability.item(), 4),
        "Final LightGBM Probability": round(final_probability.item(), 4),
    }

# Gradio Inputs
inputs = [
    gr.Slider(20, 100, step=1, label="Age"),
    gr.Slider(500, 5000, step=50, label="FVC (Forced Vital Capacity)"),
    gr.Radio(["Male", "Female"], label="Sex"),
    gr.Radio(["Ex-smoker", "Never smoked", "Currently smokes"], label="Smoking Status"),
    gr.Slider(500, 5000, step=50, label="Minimum FVC"),
    gr.Dropdown(data['Patient'].unique(), label="Patient ID"),
]

# Gradio Outputs
outputs = [
    gr.Label(label="Model Predictions"),
]

# Gradio Examples
examples = [
    [79, 2315.0, "Male", "Ex-smoker", 2315.0, "ID00007637202177411956430"],
    [65, 2500.0, "Female", "Never smoked", 2450.0, "ID00007637202177411956430"],
    [50, 1800.0, "Male", "Currently smokes", 1900.0, "ID00010637202177584929622"],
]

# Gradio Interface
gr.Interface(
    fn=predict_fibrosis,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title="Pulmonary Fibrosis Prediction",
    description="This app predicts the probability of pulmonary fibrosis based on patient metadata and CT image features.",
    theme=gr.themes.Soft(),
).launch()
