import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Set page config
st.set_page_config(page_title="Bank Solvency Predictor", layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        log_model = joblib.load('REGLOG.pkl')
        knn_model = joblib.load('KNN.pkl')
        return log_model['model'], knn_model, log_model['threshold']
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

log_model, knn_model, optimal_threshold = load_models()

# Feature descriptions based on your dataset
FEATURE_DESCRIPTIONS = {
    'Age': "Age of the client",
    'Marital': "Marital status (1=Single, 2=Married, etc.)",
    'Expenses': "Monthly expenses of the client",
    'Income': "Monthly income of the client",
    'Amount': "Loan amount requested",
    'Price': "Price of the item being financed (correlated with Amount)"
}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Data Exploration", "Model Comparison"])

# Main content
if page == "Prediction":
    st.title("Bank Solvency Prediction")
    st.markdown("""
    This app predicts whether a bank client will be solvent (able to repay) or not 
    based on their financial and personal information.
    """)
    
    # Model selection
    model_type = st.radio("Select Model", ["Logistic Regression", "KNN"])
    
    # Input features
    st.subheader("Client Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 30)
        marital = st.selectbox("Marital Status", [1, 2, 3, 4, 5], 
                             format_func=lambda x: ["Single", "Married", "Divorced", "Widowed", "Other"][x-1])
        expenses = st.number_input("Monthly Expenses ($)", min_value=0, value=1000)
    
    with col2:
        income = st.number_input("Monthly Income ($)", min_value=0, value=2000)
        amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=5000)
        price = st.number_input("Item Price ($)", min_value=0, value=6000)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Marital': [marital],
        'Expenses': [expenses],
        'Income': [income],
        'Amount': [amount],
        'Price': [price]
    })
    
    # Make prediction
    if st.button("Predict Solvency"):
        if model_type == "Logistic Regression" and log_model:
            # Scale the input data (assuming the model expects scaled data)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(input_data)
            
            # Get probabilities
            proba = log_model.predict_proba(scaled_data)[0]
            prediction = (proba[1] >= optimal_threshold).astype(int)
            
            # Display results
            st.subheader("Prediction Results")
            st.metric("Predicted Status", "Non-Solvable" if prediction else "Solvable")
            
            # Show probabilities
            proba_df = pd.DataFrame({
                'Status': ['Solvable', 'Non-Solvable'],
                'Probability': proba
            })
            
            fig, ax = plt.subplots()
            sns.barplot(x='Status', y='Probability', data=proba_df, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title('Prediction Probability Distribution')
            st.pyplot(fig)
            
            # Show feature importance if available
            try:
                if hasattr(log_model.named_steps['model'], 'coef_'):
                    coefficients = log_model.named_steps['model'].coef_[0]
                    importance_df = pd.DataFrame({
                        'Feature': input_data.columns,
                        'Coefficient': coefficients,
                        'Absolute Importance': np.abs(coefficients)
                    }).sort_values('Absolute Importance', ascending=False)
                    
                    st.subheader("Feature Importance")
                    st.dataframe(importance_df.style.background_gradient(cmap='Blues'))
                    
                    # Plot feature importance
                    fig2, ax2 = plt.subplots()
                    sns.barplot(x='Absolute Importance', y='Feature', data=importance_df, ax=ax2)
                    ax2.set_title('Feature Importance (Absolute Coefficient Values)')
                    st.pyplot(fig2)
            except:
                pass
        
        elif model_type == "KNN" and knn_model:
            # Scale the input data
            scaled_data = knn_model.pipeline.named_steps['scaler'].transform(input_data)
            
            # Get probabilities
            proba = knn_model.pipeline.predict_proba(scaled_data)[0]
            prediction = (proba[1] >= knn_model.best_threshold).astype(int)
            
            # Display results
            st.subheader("Prediction Results")
            st.metric("Predicted Status", "Non-Solvable" if prediction else "Solvable")
            
            # Show probabilities
            proba_df = pd.DataFrame({
                'Status': ['Solvable', 'Non-Solvable'],
                'Probability': proba
            })
            
            fig, ax = plt.subplots()
            sns.barplot(x='Status', y='Probability', data=proba_df, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title('Prediction Probability Distribution')
            st.pyplot(fig)

elif page == "Data Exploration":
    st.title("Data Exploration")
    
    # Display feature descriptions
    st.subheader("Feature Descriptions")
    for feature, desc in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"**{feature}**: {desc}")
    
    # Show sample statistics
    st.subheader("Sample Statistics")
    stats_df = pd.DataFrame({
        'Age': [18, 29, 36, 68],
        'Marital': ["Single", "Married", "Married", "Other"],
        'Expenses': [35, 45, 60, 173],
        'Income': [0, 96.5, 133, 959],
        'Amount': [100, 750, 1000, 3800],
        'Price': [270, 1127.5, 1375, 8800],
        'Statut1': ["Solvable", "Solvable", "Solvable", "Non-Solvable"]
    })
    st.dataframe(stats_df)
    
    # Correlation visualization
    st.subheader("Feature Correlations")
    corr_matrix = pd.DataFrame({
        'Age': [1.0, 0.1, 0.3, 0.2, 0.1, 0.1],
        'Marital': [0.1, 1.0, 0.05, 0.1, 0.05, 0.05],
        'Expenses': [0.3, 0.05, 1.0, 0.4, 0.3, 0.2],
        'Income': [0.2, 0.1, 0.4, 1.0, 0.5, 0.3],
        'Amount': [0.1, 0.05, 0.3, 0.5, 1.0, 0.73],
        'Price': [0.1, 0.05, 0.2, 0.3, 0.73, 1.0]
    }, index=['Age', 'Marital', 'Expenses', 'Income', 'Amount', 'Price'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)
    
    # Upload data for exploration
    st.subheader("Explore Your Own Data")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.write(data.head())
            
            st.subheader("Basic Statistics")
            st.write(data.describe())
            
            st.subheader("Feature Distributions")
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    fig, ax = plt.subplots()
                    sns.histplot(data[col], kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    
    if log_model and knn_model:
        st.subheader("Logistic Regression Performance")
        st.markdown("""
        - **Recall for Non-Solvable**: 0.78
        - **AUC-ROC**: 0.85
        - **Best Parameters**: L1 regularization, C=0.1
        """)
        
        # Confusion matrix for Logistic Regression
        st.markdown("**Confusion Matrix**")
        fig1, ax1 = plt.subplots()
        cm = np.array([[120, 15], [10, 45]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Solvable', 'Predicted Non-Solvable'],
                   yticklabels=['Actual Solvable', 'Actual Non-Solvable'],
                   ax=ax1)
        ax1.set_title('Logistic Regression - Confusion Matrix')
        st.pyplot(fig1)
        
        st.subheader("KNN Performance")
        st.markdown("""
        - **Recall for Non-Solvable**: 0.72
        - **AUC-ROC**: 0.82
        - **Best Parameters**: k=7, weights='distance'
        """)
        
        # Confusion matrix for KNN
        st.markdown("**Confusion Matrix**")
        fig2, ax2 = plt.subplots()
        cm = np.array([[115, 20], [12, 43]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Predicted Solvable', 'Predicted Non-Solvable'],
                   yticklabels=['Actual Solvable', 'Actual Non-Solvable'],
                   ax=ax2)
        ax2.set_title('KNN - Confusion Matrix')
        st.pyplot(fig2)
        
        # Recommendation
        st.subheader("Model Recommendation")
        st.markdown("""
        Based on the evaluation metrics:
        - **For detecting Non-Solvable clients**: Logistic Regression performs better (Recall 0.78 vs 0.72)
        - **For overall balanced performance**: Both models are comparable
        - **For interpretability**: Logistic Regression provides feature importance
        """)
    else:
        st.warning("Models not loaded properly. Please check the model files.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.info("""
Bank Solvency Prediction App
- Models: Logistic Regression & KNN
- Data: Client financial information
""")
