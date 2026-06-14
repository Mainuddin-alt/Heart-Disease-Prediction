import streamlit as st
import pandas as pd
import joblib

model= joblib.load('LogisticRegression_Heart.pkl')
scaler= joblib.load('scaler.pkl')
columns= joblib.load('columns.pkl')

st.title('Heart stroke created by Mainuddin ❤️')
st.markdown('Provided the following details')

Age	=st.slider('Age',18,100,40)
sex= st.selectbox('Sex',['M','F'])
ChestPainType= st.selectbox('ChestPainType',['ATA','NAP','ASY','TA'])
 

RestingBP= st.number_input('Resting BP (mm Hg)',80,200,120)
Cholesterol=st.number_input('Cholesterol (mg/dL)',100,600,200)
FastingBS= st.selectbox('Fasting Blood Sugar > 120 mg/dL',[0,1])

RestingECG= st.selectbox('Resting ECG',['Normal','ST','LVH'])
MaxHR	= st.slider('Max Heart Rate',50,220,150)
ExerciseAngina = st.selectbox('ExerciseAngina',['N','Y'])
Oldpeak= st.slider('Oldpeak (ST Depression)',0.0,6.0,1.0)
ST_Slope= st.selectbox('ST_Slope',['Up','Flat','Down'])

if st.button('Predict'):

    raw_input = {
        'Age': Age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'MaxHR': MaxHR,
        'Oldpeak': Oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + ChestPainType: 1,
        'RestingECG_' + RestingECG: 1,
        'ExerciseAngina_' + ExerciseAngina: 1,
        'ST_Slope_' + ST_Slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    numerical_cols = scaler.feature_names_in_

    input_df[numerical_cols] = scaler.transform(
        input_df[numerical_cols]
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")  