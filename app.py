import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fintech Fraud Detection", layout="wide")

st.title("💳 Fintech Fraud Detection System")
st.markdown("Real-time transaction risk scoring for B2B payments")

# Sidebar info
st.sidebar.header("About")
st.sidebar.markdown("""
- **Model:** RandomForest with SMOTE
- **Dataset:** Credit Card Fraud Detection
- **Challenge:** 0.17% fraud rate (imbalanced)
""")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Transaction CSV File", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head(10))
    
    # Check if 'Class' column exists
    if 'Class' not in df.columns:
        st.error("❌ Dataset must have a 'Class' column (0=Normal, 1=Fraud)")
    else:
        # Show class distribution
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(df['Class'].value_counts())
        
        with col2:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='Class', ax=ax)
            st.pyplot(fig)
        
        # Train model button
        if st.button("🚀 Train Fraud Detection Model", type="primary"):
            with st.spinner("Training model... This may take a minute"):
                # Prepare data
                X = df.drop('Class', axis=1)
                y = df['Class']
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # SMOTE
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                
                # Train
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_balanced, y_train_balanced)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                
                st.success(f"✅ Model trained! AUC-ROC: {auc:.4f}")
                
                # Show metrics
                st.subheader("📈 Model Performance")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
                st.pyplot(fig)
                
                # Store model in session for predictions
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['trained'] = True
        
        # Prediction section
        if 'trained' in st.session_state and st.session_state['trained']:
            st.markdown("---")
            st.subheader("🔮 Make Prediction")
            
            # Input form for single transaction
            with st.form("prediction_form"):
                st.write("Enter transaction details:")
                
                # Create input fields for features
                inputs = {}
                for col in df.drop('Class', axis=1).columns[:5]:  # Show first 5 features
                    inputs[col] = st.number_input(f"{col}", value=0.0)
                
                # Fill remaining with 0
                for col in df.drop('Class', axis=1).columns[5:]:
                    inputs[col] = 0.0
                
                submitted = st.form_submit_button("Predict")
                
                if submitted:
                    # Prepare input
                    input_df = pd.DataFrame([inputs])
                    input_scaled = st.session_state['scaler'].transform(input_df)
                    
                    # Predict
                    prediction = st.session_state['model'].predict(input_scaled)[0]
                    probability = st.session_state['model'].predict_proba(input_scaled)[0][1]
                    
                    # Display result
                    col1, col2 = st.columns(2)
                    col1.metric("Fraud Probability", f"{probability:.4f}")
                    
                    if prediction == 1:
                        col2.error("🚨 FRAUD DETECTED")
                    else:
                        col2.success("✅ LEGITIMATE")

else:
    st.info("👆 Please upload a CSV file to get started")
    
    # Show expected format
    st.subheader("Expected CSV Format")
    sample_data = {
        'V1': [-0.26, 0.98],
        'V2': [-0.47, -0.36],
        'V3': [2.50, 0.56],
        'Amount': [100.0, 50.0],
        'Class': [0, 1]
    }
    st.dataframe(pd.DataFrame(sample_data))
    st.caption("Note: Your CSV should have similar columns with 'Class' as target")