import gradio as gr
import pandas as pd
from fastai.vision.all import load_learner

# Load pre-trained models
resnet34_model = load_learner('./models/resnet34_model.pkl')
squeezenet_model = load_learner('./models/squeezenet1_0_model.pkl')
densenet_model = load_learner('./models/densenet121_model.pkl')

# Load the patient data
data = pd.read_csv('./data/sample_patient_data.csv')

def predict_image(patient_id, age, fvc, sex, smoking_status, min_fvc):
    """Predict pulmonary fibrosis probabilities based on image and patient metadata.

    Args:
        patient_id (str): Patient ID to fetch the associated image.
        age (float): Age of the patient.
        fvc (float): Forced Vital Capacity.
        sex (str): Male or Female.
        smoking_status (str): Smoking status.
        min_fvc (float): Minimum FVC of the patient.

    Returns:
        tuple: Predictions from each vision model and the patient image.
    """
    # Retrieve the image path for the patient
    image_path = data.loc[data['Patient'] == patient_id, 'ImagePath'].values[0]

    # Image-based predictions
    resnet_pred, _, resnet_probs = resnet34_model.predict(image_path)
    squeezenet_pred, _, squeezenet_probs = squeezenet_model.predict(image_path)
    densenet_pred, _, densenet_probs = densenet_model.predict(image_path)

    # Return predictions and the image path
    predictions = {
        "ResNet34 Probability": round(resnet_probs[1].item(), 4),
        "SqueezeNet Probability": round(squeezenet_probs[1].item(), 4),
        "DenseNet Probability": round(densenet_probs[1].item(), 4),
    }

    return predictions, image_path

# Gradio Inputs
inputs = [
    gr.Dropdown(choices=data['Patient'].unique().tolist(), label="Patient ID"),
    gr.Slider(20, 100, step=1, label="Age"),
    gr.Slider(500, 5000, step=50, label="FVC (Forced Vital Capacity)"),
    gr.Radio(["Male", "Female"], label="Sex"),
    gr.Radio(["Ex-smoker", "Never smoked", "Currently smokes"], label="Smoking Status"),
    gr.Slider(500, 5000, step=50, label="Minimum FVC"),
]

# Gradio Outputs
outputs = [
    gr.Label(label="Model Predictions"),
    gr.Image(label="Patient Image"),
]

# Gradio Examples
examples = [
    ["ID00038637202182690843176", 79, 2315.0, "Male", "Ex-smoker", 2315.0],
    ["ID00010637202177584971671", 65, 2500.0, "Female", "Never smoked", 2450.0],
    ["ID00061637202188184085559", 50, 1800.0, "Male", "Currently smokes", 1900.0],
]

# Gradio Interface
custom_theme = gr.themes.Soft(
    primary_hue="green",  # Sets the primary color to green
)

gr.Interface(
    fn=predict_image,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title="Pulmonary Fibrosis Image Prediction",
    description="This app predicts the probability of pulmonary fibrosis using pre-trained vision models and patient features. Select a patient ID and provide metadata to get predictions. The corresponding patient image will also be displayed.",
    theme=custom_theme,
).launch(share=True)
