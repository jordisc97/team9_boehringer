import gradio as gr
import pandas as pd
from fastai.vision.all import load_learner
import lightgbm as lgb

# Load the LightGBM model
lgb_model = lgb.Booster(model_file='./models/lgbm_model.txt')

# Load pre-trained models
resnet34_model = load_learner('./models/resnet34_model.pkl')
squeezenet_model = load_learner('./models/squeezenet1_0_model.pkl')
densenet_model = load_learner('./models/densenet121_model.pkl')

# Load the patient data
data = pd.read_csv('./data/sample_patient_data.csv')

def predict_image(patient_id, age, fvc, sex, smoking_status, base_week):
    """Predict pulmonary fibrosis probability based on image and patient metadata.

    Args:
        patient_id (str): Patient ID to fetch the associated image.
        age (float): Age of the patient.
        fvc (float): Forced Vital Capacity.
        sex (str): 'Male' or 'Female'.
        smoking_status (str): Smoking status.
        base_week (float): Weeks since image was taken at time 0.

    Returns:
        tuple: Final probability, risk message, and the patient image.
    """
    # Retrieve the image path for the patient
    image_path = data.loc[data['Patient'] == patient_id, 'ImagePath'].values[0]

    # Image-based predictions
    resnet_pred, _, resnet_probs = resnet34_model.predict(image_path)
    squeezenet_pred, _, squeezenet_probs = squeezenet_model.predict(image_path)
    densenet_pred, _, densenet_probs = densenet_model.predict(image_path)

    # Extract the probabilities
    resnet_prob = resnet_probs[1].item()
    squeezenet_prob = squeezenet_probs[1].item()
    densenet_prob = densenet_probs[1].item()

    # Map categorical variables to numerical values
    sex_mapping = {'Male': 1, 'Female': 2}
    smoking_status_mapping = {'Ex-smoker': 2, 'Never smoked': 0, 'Currently smokes': 5}

    sex_num = sex_mapping[sex]
    smoking_status_num = smoking_status_mapping[smoking_status]

    # Create the feature vector for the LightGBM model
    features = pd.DataFrame({
        'Age': [age],
        'FVC': [fvc],
        'Sex': [sex_num],
        'SmokingStatus': [smoking_status_num],
        'base_week': [base_week],
        'OOF_resnet34': [resnet_prob],
        'OOF_squeezenet1_0': [squeezenet_prob],
        'OOF_densenet121': [densenet_prob],
    })

    # Ensure that features only include those used by the model
    features = features[lgb_model.feature_name()]

    # Make prediction using the LightGBM model
    final_prob = lgb_model.predict(features)[0]

    # Determine risk message
    risk_message = ("Patient at High Risk: Please consider more frequent monitoring" 
                    if final_prob > 0.6 
                    else "Patient at Low Risk: Routine monitoring is sufficient"
                    )
    
    # Return final probability, risk message, and the image path
    return round(final_prob, 4), risk_message, image_path

# Gradio Inputs
inputs = [
    gr.Dropdown(choices=data['Patient'].unique().tolist(), label="Patient ID"),
    gr.Slider(20, 100, step=1, label="Age"),
    gr.Slider(500, 5000, step=50, label="FVC (Forced Vital Capacity)"),
    gr.Radio(["Male", "Female"], label="Sex"),
    gr.Radio(["Ex-smoker", "Never smoked", "Currently smokes"], label="Smoking Status"),
    gr.Number(label="Base Week"),
]

# Gradio Outputs
outputs = [
    gr.Number(label="Final Probability"),
    gr.Text(label="Risk Assessment"),
    gr.Image(label="Patient Image"),
]

# Gradio Examples
examples = [
    ["ID00038637202182690843176", 79, 2315.0, "Male", "Ex-smoker", 0],
    ["ID00010637202177584971671", 65, 2500.0, "Female", "Never smoked", 6],
    ["ID00061637202188184085559", 50, 1800.0, "Male", "Currently smokes", 5],
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
).launch(
    # share=True
    )
