import pandas as pd

MODEL_COLUMNS = [
    'Age',
    'Blood Pressure',
    'Cholesterol Level',
    'BMI',
    'Sleep Hours',
    'Triglyceride Level',
    'Fasting Blood Sugar',
    'CRP Level',
    'Homocysteine Level',
    'Gender_Male',
    'Exercise Habits_Low',
    'Exercise Habits_Medium',
    'Smoking_Yes',
    'Family Heart Disease_Yes',
    'Diabetes_Yes',
    'High Blood Pressure_Yes',
    'Low HDL Cholesterol_Yes',
    'High LDL Cholesterol_Yes',
    'Alcohol Consumption_Low',
    'Alcohol Consumption_Medium',
    'Stress Level_Low',
    'Stress Level_Medium',
    'Sugar Consumption_Low',
    'Sugar Consumption_Medium'
]


def preprocess_input(data):
    input_df = pd.DataFrame([data])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(
        columns=MODEL_COLUMNS,
        fill_value=0
    )

    return input_encoded