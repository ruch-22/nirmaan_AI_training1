import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model_performance.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary:')

# Input fields for features
rating = st.slider('Rating', 0.0, 5.0, 3.8)
company_name = st.text_input('Company Name', 'Google')
job_title = st.text_input('Job Title', 'Software Engineer')
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=3)
location = st.text_input('Location', 'Bangalore')
employment_status = st.selectbox('Employment Status', ['Full Time', 'Contract', 'Part Time', 'Internship'], index=0)
job_roles = st.text_input('Job Roles', 'Engineering')

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'Rating': [rating],
    'Company Name': [company_name],
    'Job Title': [job_title],
    'Salaries Reported': [salaries_reported],
    'Location': [location],
    'Employment Status': [employment_status],
    'Job Roles': [job_roles]
})

# Preprocess categorical features using loaded LabelEncoders
for col, encoder in label_encoders.items():
    # Handle new categories: if a category is not in the encoder's classes, assign a default value or handle it as unknown
    # For simplicity, we'll try to fit_transform, if it fails due to unknown categories,
    # we might need a more robust strategy (e.g., add to encoder, or use default like -1)
    try:
        input_data[col] = encoder.transform(input_data[col])
    except ValueError:
        # If a category is new, we need to handle it. A simple approach is to map it to a new, consistent value
        # However, for a production app, retraining the encoder with new data or a more sophisticated handling is needed.
        # For this demo, let's assume valid inputs or handle with a warning.
        st.warning(f"New category detected in '{col}'. This might affect prediction accuracy.")
        # A robust way would be to extend the encoder or assign a placeholder value.
        # For now, let's just make sure it doesn't crash by mapping new values if possible.
        # This part requires careful consideration depending on the expected input data.
        # For demonstration, we'll assign a placeholder like 0 or use the existing logic for now.
        input_data[col] = input_data[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

if st.button('Predict Salary'):
    try:
        prediction = model.predict(input_data)
        st.success(f'The predicted salary is: ₹{prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
