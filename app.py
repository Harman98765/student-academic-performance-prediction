import streamlit as st
import pandas as pd
import joblib

model=joblib.load('student_performance_model.pkl')

st.title("Student Academic Performance Predictor")
st.caption("Machine Learning project using student behavioral and academic activity data.")
st.divider()

st.write("Enter the student's information to predict their overall academic score.")

col1,col2=st.columns(2)

with col1:
    gender=st.selectbox("Gender",["female","male"])
    
with col2:
    part_time_job=st.selectbox("Part-time Job",[False,True])

col1,col2=st.columns(2)

with col1:
    absence_days=st.number_input("Absence Days",min_value=0,max_value=10,value=0)

with col2:
    extracurricular_activities=st.selectbox("Extracurricular Activities",[False,True])

col1,col2=st.columns(2)

with col1:
    weekly_self_study_hours=st.number_input("Weekly Self Study Hours",min_value=0,max_value=50,value=10)

with col2:
    career_aspiration=st.selectbox("Career Aspiration",[
        "Software Engineer",
        "Business Owner",
        "Unknown",
        "Banker",
        "Lawyer",
        "Accountant",
        "Doctor",
        "Real Estate Developer",
        "Stock Investor",
        "Construction Engineer",
        "Artist",
        "Game Developer",
        "Government Officer",
        "Teacher",
        "Designer",
        "Scientist",
        "Writer"
    ])

input_data=pd.DataFrame({
    'gender':[gender],
    'part_time_job':[part_time_job],
    'absence_days':[absence_days],
    'extracurricular_activities':[extracurricular_activities],
    'weekly_self_study_hours':[weekly_self_study_hours],
    'career_aspiration':[career_aspiration]
})

if st.button("Predict Performance"):
    prediction=model.predict(input_data)[0]

    if prediction>=85:
        performance="High"
    elif prediction>=75:
        performance="Moderate"
    else:
        performance="Lower"

    st.success(f"Predicted Overall Score: {prediction:.2f} / 100")
    st.info(f"Predicted Performance Band: {performance}")
    st.progress(min(prediction/100,1.0))

st.divider()
st.subheader("About the Model")
st.write("This application uses a Gradient Boosting regression model to predict a student's overall academic score.")
st.write("The model was trained using student study habits, attendance-related information, extracurricular activities, gender, part-time job status, and career aspiration.")

st.caption("Model performance on the held-out test set: MAE ≈ 3.66 | RMSE ≈ 4.74 | R² ≈ 0.477")
