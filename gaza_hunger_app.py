

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import pickle
from datetime import datetime

# Sklearn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, accuracy_score
)

warnings.filterwarnings('ignore')

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Gaza Hunger Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    h3 {
        color: #34495e;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    .prediction-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-moderate {
        background-color: #ff9800;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-low {
        background-color: #4caf50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================================================================
# DATA LOADING AND CACHING
# ================================================================================

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_excel('GazaHungerData.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_feature_descriptions():
    """Get descriptions for all features in the dataset"""
    descriptions = {
        'Q1-Q7': 'Food Access Questions (Yes/No)',
        'Q8-Q10': 'Frequency Questions (Rarely/Sometimes/Mostly)',
        'Q11-Q28': 'Household Food Security Indicators',
        'Q29': 'Number of family members',
        'Q30': 'Having a chronic disease',
        'Q31': 'Times displaced',
        'Q32': 'Number of children',
        'Q33': 'Having children under 5',
        'Q34': 'Having pregnant women',
        'Q35': 'Number of meals',
        'Q36': 'Income sufficiency',
        'Q37': 'Number of elderly people',
        'Q38': 'Having disabled members',
        'Q39': 'Number of women',
        'Q40': 'Number of men',
        'Q41': 'Number of students',
        'Q42': 'Total family income',
        'Q43': 'Marital status',
        'Q44': 'Age',
        'Q45': 'Work situation',
        'Q46': 'Education level',
        'Q47': 'House destruction status',
        'Q48': 'Living area type',
        'Q49': 'Current shelter type',
        'Q50': 'Water availability (Target Variable)'
    }
    return descriptions

# ================================================================================
# DATA PREPROCESSING FUNCTIONS
# ================================================================================

def preprocess_data(df):
    """
    Preprocess the dataset: encode categorical variables and scale features
    
    Args:
        df: Raw dataframe
    
    Returns:
        X: Feature matrix
        y: Target variable
        label_encoders: Dictionary of label encoders for each column
        scaler: StandardScaler object
    """
    df_processed = df.copy()
    
    # Initialize label encoders dictionary
    label_encoders = {}
    
    # Encode categorical columns
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Separate features and target
    X = df_processed.drop('Q50', axis=1)
    y = df_processed['Q50']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, label_encoders, scaler

# ================================================================================
# MODEL TRAINING FUNCTIONS
# ================================================================================

@st.cache_resource
def train_models(_X_train, _y_train):
    """
    Train multiple models and return them
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Logistic Regression
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    )
    models['Logistic Regression'].fit(_X_train, _y_train)
    
    # Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=10
    )
    models['Random Forest'].fit(_X_train, _y_train)
    
    # Gradient Boosting
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        learning_rate=0.1
    )
    models['Gradient Boosting'].fit(_X_train, _y_train)
    
    return models

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred
    }

# ================================================================================
# PREDICTION FUNCTIONS
# ================================================================================

def predict_one_household(model, household_data, feature_names, scaler):
    """
    Predict hunger severity for a single household
    
    Args:
        model: Trained model
        household_data: Dictionary of household features
        feature_names: List of feature names
        scaler: Fitted StandardScaler
    
    Returns:
        Dictionary containing prediction results
    """
    # Create dataframe from household data
    household_df = pd.DataFrame([household_data])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in household_df.columns:
            household_df[feature] = 0
    
    # Reorder columns to match training data
    household_df = household_df[feature_names]
    
    # Scale the data
    household_scaled = scaler.transform(household_df)
    
    # Make prediction
    prediction = model.predict(household_scaled)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(household_scaled)[0]
    else:
        probabilities = None
    
    return {
        'prediction': prediction,
        'probabilities': probabilities
    }

# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def plot_confusion_matrix(cm, class_names):
    """Create an interactive confusion matrix plot"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        width=600
    )
    
    return fig

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Most Important Features',
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature Name'},
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=600, showlegend=False)
        return fig
    else:
        return None

def plot_class_distribution(y, title="Class Distribution"):
    """Plot the distribution of classes"""
    value_counts = pd.Series(y).value_counts()
    
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def plot_model_comparison(results_dict):
    """Compare performance across different models"""
    metrics_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': [results_dict[m]['accuracy'] for m in results_dict.keys()],
        'Precision': [results_dict[m]['precision'] for m in results_dict.keys()],
        'Recall': [results_dict[m]['recall'] for m in results_dict.keys()],
        'F1-Score': [results_dict[m]['f1_score'] for m in results_dict.keys()]
    })
    
    fig = go.Figure()
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            text=metrics_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    # Header
    st.title("🌾 Gaza Hunger Prediction System")
    st.markdown("""
    **Predictive Analytics for Household Hunger Severity Assessment**
    
    This system uses machine learning to predict household hunger severity based on 
    socioeconomic indicators, displacement patterns, and food security metrics.
    
    ---
    """)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Gaza+Hunger+Analytics", use_container_width=True)
        st.markdown("### 📊 Navigation")
        st.markdown("Use the tabs above to navigate through different sections of the application.")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Dataset:** 1,209 households from Gaza Strip
        
        **Period:** May-July 2024
        
        **Models:** Logistic Regression, Random Forest, Gradient Boosting
        
        **Target:** Water Availability (Proxy for Hunger Severity)
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"],
            help="Choose the machine learning model for predictions"
        )
        
        # Test size
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data file.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🔧 Preprocessing",
        "🤖 Model Training",
        "📈 Model Performance",
        "🎯 Make Predictions",
        "📥 Export Results"
    ])
    
    # ============================================================================
    # TAB 1: DATA OVERVIEW
    # ============================================================================
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Households", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Categorical Features", df.select_dtypes(include='object').shape[1])
        with col4:
            st.metric("Numerical Features", df.select_dtypes(include=['int64', 'float64']).shape[1])
        
        st.markdown("---")
        
        # Dataset preview
        st.subheader("📋 Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Feature descriptions
        st.subheader("📝 Feature Descriptions")
        descriptions = get_feature_descriptions()
        
        desc_df = pd.DataFrame(list(descriptions.items()), columns=['Feature Range', 'Description'])
        st.table(desc_df)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Features**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**Categorical Features**")
            categorical_summary = []
            for col in df.select_dtypes(include='object').columns:
                unique_count = df[col].nunique()
                most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
                categorical_summary.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Most Common': most_common
                })
            
            if categorical_summary:
                st.dataframe(pd.DataFrame(categorical_summary), use_container_width=True)
        
        # Target variable distribution
        st.subheader("🎯 Target Variable Distribution (Q50 - Water Availability)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_class_distribution(df['Q50'], "Water Availability Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Class Counts**")
            value_counts = df['Q50'].value_counts()
            for idx, count in value_counts.items():
                st.metric(str(idx), count)
        
        # Missing values
        st.subheader("🔍 Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title='Missing Values by Feature',
                labels={'x': 'Count', 'y': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
    
    # ============================================================================
    # TAB 2: PREPROCESSING
    # ============================================================================
    with tab2:
        st.header("🔧 Data Preprocessing")
        
        st.markdown("""
        This section handles the transformation of raw data into a format suitable for machine learning:
        
        1. **Label Encoding**: Converting categorical variables to numerical values
        2. **Feature Scaling**: Standardizing features to have mean=0 and std=1
        3. **Train-Test Split**: Dividing data into training and testing sets
        """)
        
        if st.button("🚀 Start Preprocessing", type="primary"):
            with st.spinner("Processing data..."):
                # Preprocess data
                X, y, label_encoders, scaler = preprocess_data(df)
                
                # Store in session state
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['label_encoders'] = label_encoders
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = X.columns.tolist()
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
                
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                
                st.success("✅ Preprocessing completed successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Processed Data Summary")
                    st.metric("Training Samples", len(X_train))
                    st.metric("Testing Samples", len(X_test))
                    st.metric("Total Features", X.shape[1])
                
                with col2:
                    st.subheader("🏷️ Encoded Categories")
                    st.write(f"Total categorical columns encoded: {len(label_encoders)}")
                    
                    # Show sample encoding
                    if 'Q50' in label_encoders:
                        st.markdown("**Target Variable Encoding (Q50):**")
                        target_encoder = label_encoders['Q50']
                        encoding_dict = {original: encoded for encoded, original in enumerate(target_encoder.classes_)}
                        st.json(encoding_dict)
                
                # Show processed data preview
                st.subheader("📋 Processed Data Preview")
                st.dataframe(X.head(10), use_container_width=True)
        
        else:
            st.info("👆 Click the button above to start preprocessing the data.")
    
    # ============================================================================
    # TAB 3: MODEL TRAINING
    # ============================================================================
    with tab3:
        st.header("🤖 Model Training")
        
        if 'X_train' not in st.session_state:
            st.warning("⚠️ Please complete the preprocessing step first!")
            return
        
        st.markdown("""
        Train multiple machine learning models and compare their performance:
        
        - **Logistic Regression**: Linear model for classification
        - **Random Forest**: Ensemble of decision trees
        - **Gradient Boosting**: Sequential ensemble method
        """)
        
        if st.button("🎯 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a moment."):
                # Get training data
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                # Train models
                models = train_models(X_train, y_train)
                st.session_state['models'] = models
                
                # Evaluate all models
                results = {}
                for model_name, model in models.items():
                    results[model_name] = evaluate_model(model, X_test, y_test)
                
                st.session_state['results'] = results
                
                st.success("✅ All models trained successfully!")
                
                # Display comparison
                st.subheader("📊 Model Performance Comparison")
                
                fig = plot_model_comparison(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("📈 Detailed Metrics")
                
                metrics_data = []
                for model_name, result in results.items():
                    metrics_data.append({
                        'Model': model_name,
                        'Accuracy': f"{result['accuracy']:.4f}",
                        'Precision': f"{result['precision']:.4f}",
                        'Recall': f"{result['recall']:.4f}",
                        'F1-Score': f"{result['f1_score']:.4f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Best model
                best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                st.success(f"🏆 Best performing model: **{best_model[0]}** with accuracy: **{best_model[1]['accuracy']:.4f}**")
        
        else:
            st.info("👆 Click the button above to train all models.")
    
    # ============================================================================
    # TAB 4: MODEL PERFORMANCE
    # ============================================================================
    with tab4:
        st.header("📈 Model Performance Analysis")
        
        if 'models' not in st.session_state or 'results' not in st.session_state:
            st.warning("⚠️ Please train the models first!")
            return
        
        # Model selection for detailed analysis
        selected_model = st.selectbox(
            "Select a model for detailed analysis",
            list(st.session_state['models'].keys()),
            index=list(st.session_state['models'].keys()).index(model_choice) if model_choice in st.session_state['models'].keys() else 0
        )
        
        result = st.session_state['results'][selected_model]
        model = st.session_state['models'][selected_model]
        
        # Metrics overview
        st.subheader(f"📊 {selected_model} - Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{result['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{result['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{result['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{result['f1_score']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Confusion Matrix")
            
            # Get class names
            if 'label_encoders' in st.session_state and 'Q50' in st.session_state['label_encoders']:
                class_names = st.session_state['label_encoders']['Q50'].classes_
            else:
                class_names = [f"Class {i}" for i in range(len(result['confusion_matrix']))]
            
            fig_cm = plot_confusion_matrix(result['confusion_matrix'], class_names)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.subheader("📋 Classification Report")
            
            # Display classification report
            report_df = pd.DataFrame(result['classification_report']).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
        
        # Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.subheader("🔍 Feature Importance Analysis")
            
            top_n = st.slider("Number of top features to display", 10, 30, 20)
            
            fig_importance = plot_feature_importance(
                model,
                st.session_state['feature_names'],
                top_n=top_n
            )
            
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Download feature importance
                importance_df = pd.DataFrame({
                    'Feature': st.session_state['feature_names'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                csv = importance_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Feature Importance CSV",
                    data=csv,
                    file_name=f"feature_importance_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
    
    # ============================================================================
    # TAB 5: MAKE PREDICTIONS
    # ============================================================================
    with tab5:
        st.header("🎯 Single Household Prediction")
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train the models first!")
            return
        
        st.markdown("""
        Enter household information to predict hunger severity. All fields are required for accurate predictions.
        """)
        
        # Get the selected model
        selected_prediction_model = st.session_state['models'][model_choice]
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("🏠 Household Information")
            
            col1, col2, col3 = st.columns(3)
            
            household_data = {}
            
            with col1:
                st.markdown("**👨‍👩‍👧‍👦 Family Composition**")
                household_data['Q29'] = st.number_input("Number of family members", min_value=1, max_value=20, value=5)
                household_data['Q32'] = st.number_input("Number of children", min_value=0, max_value=15, value=2)
                household_data['Q37'] = st.number_input("Number of elderly people", min_value=0, max_value=10, value=0)
                household_data['Q39'] = st.number_input("Number of women", min_value=0, max_value=15, value=2)
                household_data['Q40'] = st.number_input("Number of men", min_value=0, max_value=15, value=2)
                household_data['Q41'] = st.number_input("Number of students", min_value=0, max_value=15, value=1)
            
            with col2:
                st.markdown("**💰 Economic Situation**")
                household_data['Q42'] = st.number_input("Total family income (ILS)", min_value=0, max_value=10000, value=1000)
                household_data['Q44'] = st.number_input("Head of household age", min_value=18, max_value=100, value=35)
                household_data['Q35'] = st.number_input("Number of meals per day", min_value=0, max_value=5, value=2)
                
                # Encode Yes/No questions
                household_data['Q30'] = 1 if st.selectbox("Having a chronic disease?", ["No", "Yes"]) == "Yes" else 0
                household_data['Q33'] = 1 if st.selectbox("Having children under 5?", ["No", "Yes"]) == "Yes" else 0
                household_data['Q34'] = 1 if st.selectbox("Having pregnant women?", ["No", "Yes"]) == "Yes" else 0
                household_data['Q38'] = 1 if st.selectbox("Having disabled members?", ["No", "Yes"]) == "Yes" else 0
                household_data['Q45'] = 1 if st.selectbox("Currently working?", ["No", "Yes"]) == "Yes" else 0
            
            with col3:
                st.markdown("**🏚️ Living Conditions**")
                household_data['Q31'] = st.number_input("Times displaced", min_value=0, max_value=20, value=2)
                
                # These would need proper encoding based on your label encoders
                shelter_options = ["House", "Tent", "School", "Apartment"]
                area_options = ["City", "Camp", "Village"]
                destruction_options = ["NoDestruction", "PartDestruction", "ComplDestruction"]
                education_options = ["None", "Primary", "Secondary", "Bachelor", "Master"]
                marital_options = ["Single", "Married", "Divorced", "Widowed"]
                income_suff_options = ["NotSufficient", "SomewhatSufficient", "Sufficient"]
                
                shelter = st.selectbox("Current shelter type", shelter_options)
                area = st.selectbox("Living area type", area_options)
                destruction = st.selectbox("House destruction status", destruction_options)
                education = st.selectbox("Education level", education_options)
                marital = st.selectbox("Marital status", marital_options)
                income_suff = st.selectbox("Income sufficiency", income_suff_options)
                
                # Encode these selections
                household_data['Q49'] = shelter_options.index(shelter)
                household_data['Q48'] = area_options.index(area)
                household_data['Q47'] = destruction_options.index(destruction)
                household_data['Q46'] = education_options.index(education)
                household_data['Q43'] = marital_options.index(marital)
                household_data['Q36'] = income_suff_options.index(income_suff)
            
            st.markdown("---")
            st.markdown("**🍽️ Food Access Questions**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                for i in range(1, 8):
                    household_data[f'Q{i}'] = 1 if st.selectbox(f"Q{i}: Food access indicator", ["No", "Yes"], key=f"q{i}") == "Yes" else 0
            
            with col2:
                freq_options = ["Rarely", "Sometimes", "Mostly"]
                for i in range(8, 11):
                    freq = st.selectbox(f"Q{i}: Frequency indicator", freq_options, key=f"q{i}")
                    household_data[f'Q{i}'] = freq_options.index(freq)
            
            # Additional questions (Q11-Q28)
            st.markdown("**📋 Additional Food Security Indicators**")
            cols = st.columns(6)
            for i, q in enumerate(range(11, 29)):
                col_idx = i % 6
                with cols[col_idx]:
                    household_data[f'Q{q}'] = 1 if st.checkbox(f"Q{q}", key=f"q{q}") else 0
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Hunger Severity", type="primary")
            
            if submitted:
                with st.spinner("Making prediction..."):
                    # Make prediction
                    result = predict_one_household(
                        selected_prediction_model,
                        household_data,
                        st.session_state['feature_names'],
                        st.session_state['scaler']
                    )
                    
                    prediction = result['prediction']
                    probabilities = result['probabilities']
                    
                    # Get class name
                    if 'label_encoders' in st.session_state and 'Q50' in st.session_state['label_encoders']:
                        class_name = st.session_state['label_encoders']['Q50'].inverse_transform([prediction])[0]
                    else:
                        class_name = f"Class {prediction}"
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("🎯 Prediction Result")
                    
                    # Determine severity level for styling
                    severity_map = {
                        'No': 'low',
                        'Intermittently': 'moderate',
                        'Regularly': 'high',
                        'Yes Intermittently': 'high'
                    }
                    
                    severity = severity_map.get(class_name, 'moderate')
                    
                    if severity == 'high':
                        st.markdown(f'<div class="prediction-high">⚠️ HIGH RISK: {class_name}</div>', unsafe_allow_html=True)
                        st.error("**Immediate intervention recommended!** This household shows signs of severe food insecurity.")
                    elif severity == 'moderate':
                        st.markdown(f'<div class="prediction-moderate">⚡ MODERATE RISK: {class_name}</div>', unsafe_allow_html=True)
                        st.warning("**Monitoring required.** This household may need support in the near future.")
                    else:
                        st.markdown(f'<div class="prediction-low">✅ LOW RISK: {class_name}</div>', unsafe_allow_html=True)
                        st.success("This household appears to have stable food security.")
                    
                    # Show probabilities if available
                    if probabilities is not None:
                        st.markdown("---")
                        st.subheader("📊 Prediction Probabilities")
                        
                        prob_df = pd.DataFrame({
                            'Class': st.session_state['label_encoders']['Q50'].classes_,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        fig = px.bar(
                            prob_df,
                            x='Class',
                            y='Probability',
                            title='Probability Distribution',
                            color='Probability',
                            color_continuous_scale='RdYlGn_r'
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show table
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(prob_df, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommendations")
                    
                    if severity == 'high':
                        st.markdown("""
                        1. **Immediate food assistance** - Prioritize for emergency food parcels
                        2. **Medical screening** - Check for malnutrition, especially in children and elderly
                        3. **Livelihood support** - Connect with employment or income generation programs
                        4. **Follow-up visit** - Schedule within 1 week
                        """)
                    elif severity == 'moderate':
                        st.markdown("""
                        1. **Regular monitoring** - Include in monthly assessment schedule
                        2. **Food vouchers** - Consider for supplementary food assistance
                        3. **Education programs** - Nutrition education for household members
                        4. **Follow-up visit** - Schedule within 1 month
                        """)
                    else:
                        st.markdown("""
                        1. **Routine monitoring** - Include in quarterly assessment
                        2. **Preventive education** - Provide information on food storage and nutrition
                        3. **Economic opportunities** - Share information about available programs
                        """)
    
    # ============================================================================
    # TAB 6: EXPORT RESULTS
    # ============================================================================
    with tab6:
        st.header("📥 Export Results and Data")
        
        st.markdown("""
        Download processed data, model predictions, and performance metrics for further analysis.
        """)
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train the models first!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Processed Dataset")
            
            if st.button("📥 Download Processed Data", key="download_processed"):
                if 'X' in st.session_state and 'y' in st.session_state:
                    # Combine X and y
                    export_df = st.session_state['X'].copy()
                    export_df['Target'] = st.session_state['y']
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"gaza_hunger_processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Processed data not available. Please complete preprocessing first.")
        
        with col2:
            st.subheader("🎯 Model Predictions")
            
            if st.button("📥 Download Predictions", key="download_predictions"):
                if 'results' in st.session_state and 'X_test' in st.session_state:
                    selected_model_name = model_choice
                    result = st.session_state['results'][selected_model_name]
                    
                    # Create predictions dataframe
                    predictions_df = st.session_state['X_test'].copy()
                    predictions_df['Actual'] = st.session_state['y_test'].values
                    predictions_df['Predicted'] = result['predictions']
                    
                    # Decode if possible
                    if 'label_encoders' in st.session_state and 'Q50' in st.session_state['label_encoders']:
                        encoder = st.session_state['label_encoders']['Q50']
                        predictions_df['Actual_Label'] = encoder.inverse_transform(predictions_df['Actual'])
                        predictions_df['Predicted_Label'] = encoder.inverse_transform(predictions_df['Predicted'])
                    
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"gaza_hunger_predictions_{selected_model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Predictions not available. Please train models first.")
        
        st.markdown("---")
        
        # Model Performance Report
        st.subheader("📈 Model Performance Report")
        
        if st.button("📥 Generate Performance Report", key="download_report"):
            if 'results' in st.session_state:
                # Create comprehensive report
                report_lines = []
                report_lines.append("=" * 80)
                report_lines.append("GAZA HUNGER PREDICTION - MODEL PERFORMANCE REPORT")
                report_lines.append("=" * 80)
                report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"\nDataset Size: {df.shape[0]} households")
                report_lines.append(f"Number of Features: {df.shape[1] - 1}")
                report_lines.append(f"Test Set Size: {test_size}%")
                report_lines.append("\n" + "=" * 80)
                
                for model_name, result in st.session_state['results'].items():
                    report_lines.append(f"\n\nMODEL: {model_name}")
                    report_lines.append("-" * 80)
                    report_lines.append(f"Accuracy:  {result['accuracy']:.4f}")
                    report_lines.append(f"Precision: {result['precision']:.4f}")
                    report_lines.append(f"Recall:    {result['recall']:.4f}")
                    report_lines.append(f"F1-Score:  {result['f1_score']:.4f}")
                    
                    report_lines.append("\n\nConfusion Matrix:")
                    cm = result['confusion_matrix']
                    for row in cm:
                        report_lines.append("  " + "  ".join(f"{val:5d}" for val in row))
                    
                    report_lines.append("\n\nClassification Report:")
                    report_df = pd.DataFrame(result['classification_report']).transpose()
                    report_lines.append(report_df.to_string())
                
                report_lines.append("\n\n" + "=" * 80)
                report_lines.append("END OF REPORT")
                report_lines.append("=" * 80)
                
                report_text = "\n".join(report_lines)
                
                st.download_button(
                    label="💾 Download Report (TXT)",
                    data=report_text,
                    file_name=f"gaza_hunger_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.error("Performance metrics not available. Please train models first.")
        
        # Export trained model
        st.markdown("---")
        st.subheader("🤖 Export Trained Model")
        
        st.info("""
        Export the trained model as a pickle file for deployment or future use.
        This allows you to use the model in other applications without retraining.
        """)
        
        if st.button("📥 Export Model", key="export_model"):
            if 'models' in st.session_state:
                selected_export_model = st.session_state['models'][model_choice]
                
                # Serialize model
                model_bytes = pickle.dumps(selected_export_model)
                
                st.download_button(
                    label="💾 Download Model (.pkl)",
                    data=model_bytes,
                    file_name=f"gaza_hunger_model_{model_choice.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
                
                st.success(f"✅ Model '{model_choice}' is ready for download!")
            else:
                st.error("No trained models available. Please train models first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>Gaza Hunger Prediction System v1.0</strong></p>
        <p>Developed for humanitarian crisis response | February 2026</p>
        <p>⚠️ For official use only - Results should be validated by field workers</p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================================
# RUN APPLICATION
# ================================================================================

if __name__ == "__main__":
    main()