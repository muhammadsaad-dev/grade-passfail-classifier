import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

st.title("🎓 Student Grade Prediction Form")

# --- Define all valid categories for binary columns ---
BINARY_CATEGORIES = {
    'school': ['GP', 'MS'],
    'sex': ['F', 'M'],
    'address': ['U', 'R'],
    'famsize': ['LE3', 'GT3'],
    'Pstatus': ['T', 'A'],
    'schoolsup': ['yes', 'no'],
    'famsup': ['yes', 'no'],
    'paid': ['yes', 'no'],
    'activities': ['yes', 'no'],
    'nursery': ['yes', 'no'],
    'higher': ['yes', 'no'],
    'internet': ['yes', 'no'],
    'romantic': ['yes', 'no'],
    'subject': ['Math', 'Portuguese']
}

# --- Pre-fitted encoder class ---
class BinaryLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, binary_cols):
        self.binary_cols = binary_cols
        self.encoders = {}
        
        # Initialize encoders with all known categories
        for col in binary_cols:
            le = LabelEncoder()
            le.fit(BINARY_CATEGORIES[col])
            self.encoders[col] = le

    def fit(self, X, y=None):
        # Already fitted during init
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.binary_cols:
            # Convert to string and handle missing/unknown values
            X[col] = X[col].astype(str)
            unknown_mask = ~X[col].isin(self.encoders[col].classes_)
            if unknown_mask.any():
                default_value = self.encoders[col].classes_[0]
                X.loc[unknown_mask, col] = default_value
            X[col] = self.encoders[col].transform(X[col])
        return X

# --- Feature Engineering ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
        
    def transform(self, X):
        X = X.copy()
        X['change'] = X['G2'] - X['G1']
        X['G'] = (X['G1'] + X['G2']) / 2
        X['Pedu'] = (X['Medu'] + X['Fedu']) / 2
        return X

# --- Form ---
with st.form("student_form"):
    st.header("🧍 General Information")
    school = st.selectbox("School", BINARY_CATEGORIES['school'])
    sex = st.selectbox("Sex", BINARY_CATEGORIES['sex'])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Home Address", BINARY_CATEGORIES['address'])
    famsize = st.selectbox("Family Size", BINARY_CATEGORIES['famsize'])
    Pstatus = st.selectbox("Parental Cohabitation Status", BINARY_CATEGORIES['Pstatus'])

    st.header("🎓 Parental Info")
    Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])
    Mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
    Fjob = st.selectbox("Father's Job", ['teacher', 'health', 'services', 'at_home', 'other'])

    st.header("🏫 School-related Info")
    reason = st.selectbox("Reason for Choosing School", ['home', 'reputation', 'course', 'other'])
    guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
    traveltime = st.selectbox("Travel Time to School", [1, 2, 3, 4])
    studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
    failures = st.number_input("Past Class Failures", 0, 4, 0)

    st.header("📚 Support & Activities")
    schoolsup = st.selectbox("School Support", BINARY_CATEGORIES['schoolsup'])
    famsup = st.selectbox("Family Support", BINARY_CATEGORIES['famsup'])
    paid = st.selectbox("Extra Paid Classes", BINARY_CATEGORIES['paid'])
    activities = st.selectbox("Extra-curricular Activities", BINARY_CATEGORIES['activities'])
    nursery = st.selectbox("Attended Nursery School", BINARY_CATEGORIES['nursery'])
    higher = st.selectbox("Wants Higher Education", BINARY_CATEGORIES['higher'])
    internet = st.selectbox("Internet Access at Home", BINARY_CATEGORIES['internet'])
    romantic = st.selectbox("In a Romantic Relationship", BINARY_CATEGORIES['romantic'])

    st.header("📊 Behavior & Grades")
    famrel = st.slider("Family Relationship Quality", 1, 5, 4)
    freetime = st.slider("Free Time After School", 1, 5, 3)
    goout = st.slider("Going Out with Friends", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 2)
    health = st.slider("Current Health Status", 1, 5, 5)
    absences = st.number_input("Number of School Absences", 0, 93, 0)

    G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
    G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
    subject = st.selectbox("Subject", BINARY_CATEGORIES['subject'])

    submitted = st.form_submit_button("Submit")

# --- On Submit ---
if submitted:
    # Collect input data
    input_data = {
        'school': school, 'sex': sex, 'age': age, 'address': address,
        'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
        'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
        'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
        'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid, 'activities': activities,
        'nursery': nursery, 'higher': higher, 'internet': internet, 'romantic': romantic,
        'famrel': famrel, 'freetime': freetime, 'goout': goout,
        'Dalc': Dalc, 'Walc': Walc, 'health': health, 'absences': absences,
        'G1': G1, 'G2': G2, 'subject': subject
    }

    df = pd.DataFrame([input_data])
    st.subheader("📄 Submitted Data")
    st.dataframe(df)

    # --- Feature Engineering ---
    df = FeatureEngineer().transform(df)

    # --- Encoding ---
    encoder = BinaryLabelEncoder(binary_cols=list(BINARY_CATEGORIES.keys()))
    try:
        df = encoder.transform(df)
    except Exception as e:
        st.error(f"Error during encoding: {str(e)}")
        st.stop()

    # --- Load Model & Predict ---
    try:
        model = joblib.load("grade_classification_model.pkl")
        prediction = model.predict_proba(df)
        st.success(f"🎯 Passing Probability: {prediction[0][1]}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")