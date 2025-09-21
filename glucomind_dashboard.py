# ============================================
# GLUCOMIND STREAMLIT DASHBOARD
# Save as: glucomind_dashboard.py
# Run with: streamlit run glucomind_dashboard.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import tensorflow (may not be needed for all models)
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not available. Neural network models won't work.")

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="GlucoMind - AI Glucose Monitoring",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    .stAlert {padding: 1rem; border-radius: 0.5rem;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        text-align: center;
        padding: 2rem;
        background-color: #f0f2f6;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND CONFIGURATION
# ============================================
@st.cache_resource
def load_models():
    """Load the saved best model and scalers"""
    try:
        # Load metadata
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata['best_model']
        st.sidebar.info(f"📊 Loaded Model: {model_type}")
        
        # Load scalers
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        
        # Load the model based on type
        if model_type in ['BiLSTM', 'LSTM']:
            # For neural networks, check if H5 file exists
            if model_type == 'BiLSTM' and TF_AVAILABLE:
                try:
                    model = load_model('best_model_bilstm.h5', compile=False)
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                except:
                    # Try loading from pkl wrapper
                    with open('best_model.pkl', 'rb') as f:
                        model_wrapper = pickle.load(f)
                    if 'model_path' in model_wrapper:
                        model = load_model(model_wrapper['model_path'], compile=False)
                        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    else:
                        st.error("Neural network model file not found")
                        return None, None, None, None, None
            elif model_type == 'LSTM' and TF_AVAILABLE:
                try:
                    model = load_model('best_model_lstm.h5', compile=False)
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                except:
                    with open('best_model.pkl', 'rb') as f:
                        model_wrapper = pickle.load(f)
                    if 'model_path' in model_wrapper:
                        model = load_model(model_wrapper['model_path'], compile=False)
                        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    else:
                        st.error("Neural network model file not found")
                        return None, None, None, None, None
            else:
                st.error("TensorFlow not available for neural network models")
                return None, None, None, None, None
        else:
            # For traditional ML models (Linear Regression, ARIMA)
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Handle ARIMA wrapper
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
        
        return model, scaler_X, scaler_y, metadata, model_type
        
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        st.info("Please ensure these files are in the same directory: best_model.pkl, scaler_X.pkl, scaler_y.pkl, model_metadata.json")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_glucose(model, scaler_X, scaler_y, input_features, model_type):
    """Make glucose prediction using the loaded model"""
    try:
        # Scale input features
        input_scaled = scaler_X.transform(input_features.reshape(1, -1))
        
        if model_type in ['BiLSTM', 'LSTM']:
            # Reshape for LSTM (samples, timesteps, features)
            input_lstm = input_scaled.reshape(1, 1, input_scaled.shape[1])
            
            # Make prediction
            prediction_scaled = model.predict(input_lstm, verbose=0)
            
            # Inverse transform to get actual glucose value
            prediction = scaler_y.inverse_transform(prediction_scaled)
            return prediction[0][0]
            
        elif model_type == 'Linear Regression':
            # Direct prediction for sklearn models
            prediction = model.predict(input_scaled)
            return prediction[0]
            
        elif model_type == 'ARIMA':
            # ARIMA forecast
            if hasattr(model, 'forecast'):
                prediction = model.forecast(steps=1)
                return prediction[0] if hasattr(prediction, '__len__') else prediction
            else:
                st.error("ARIMA model doesn't have forecast method")
                return None
        else:
            st.error(f"Unknown model type: {model_type}")
            return None
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ============================================
# MAIN DASHBOARD
# ============================================
def main():
    # Header
    st.title("🩺 GlucoMind - Smart Glucose Monitoring Dashboard")
    st.markdown("### Glucose Prediction System")
    st.markdown("---")
    
    # Load model
    model, scaler_X, scaler_y, metadata, model_type = load_models()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check the error messages above.")
        st.stop()
    
    # Display model info
    with st.expander("📊 Model Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Type", model_type)
        with col2:
            st.metric("R² Score", f"{metadata['performance']['R2']:.4f}")
        with col3:
            st.metric("RMSE", f"{metadata['performance']['RMSE']:.2f} mg/dL")
        with col4:
            st.metric("MAE", f"{metadata['performance']['MAE']:.2f} mg/dL")
    
    # ============================================
    # SIDEBAR - USER INPUT
    # ============================================
    st.sidebar.header("📝 Input Features")
    st.sidebar.markdown("Enter values for glucose prediction:")
    
    # Get feature names from metadata
    feature_names = metadata['feature_columns']
    
    # Create input fields dynamically
    user_inputs = {}
    
    # Define default values and descriptions for each feature
    feature_configs = {
        'glucose_lag_1': {
            'label': "Previous Glucose (5 min ago)",
            'min': 40.0, 'max': 400.0, 'default': 120.0, 'step': 1.0,
            'help': "Your glucose level 5 minutes ago"
        },
        'glucose_lag_3': {
            'label': "Glucose (15 min ago)",
            'min': 40.0, 'max': 400.0, 'default': 118.0, 'step': 1.0,
            'help': "Your glucose level 15 minutes ago"
        },
        'glucose_lag_6': {
            'label': "Glucose (30 min ago)",
            'min': 40.0, 'max': 400.0, 'default': 115.0, 'step': 1.0,
            'help': "Your glucose level 30 minutes ago"
        },
        'glucose_roll_mean_1h': {
            'label': "1-Hour Average Glucose",
            'min': 40.0, 'max': 400.0, 'default': 117.0, 'step': 1.0,
            'help': "Average glucose over the past hour"
        },
        'insulin_bolus': {
            'label': "Insulin Bolus (units)",
            'min': 0.0, 'max': 20.0, 'default': 0.0, 'step': 0.5,
            'help': "Rapid-acting insulin dose"
        },
        'insulin_basal': {
            'label': "Insulin Basal (units/hr)",
            'min': 0.0, 'max': 5.0, 'default': 1.0, 'step': 0.1,
            'help': "Basal insulin rate"
        },
        'carbs': {
            'label': "Carbohydrates (grams)",
            'min': 0.0, 'max': 150.0, 'default': 0.0, 'step': 5.0,
            'help': "Carbs consumed recently"
        }
    }
    
    # Create input fields for available features
    for feature in feature_names:
        if feature in feature_configs:
            config = feature_configs[feature]
            user_inputs[feature] = st.sidebar.number_input(
                config['label'],
                min_value=config['min'],
                max_value=config['max'],
                value=config['default'],
                step=config['step'],
                help=config['help']
            )
        elif feature == 'hour':
            user_inputs['hour'] = st.sidebar.slider(
                "Hour of Day",
                min_value=0, max_value=23,
                value=datetime.now().hour,
                help="Current hour (24-hour format)"
            )
        elif feature == 'day_of_week':
            user_inputs['day_of_week'] = st.sidebar.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
                index=datetime.now().weekday()
            )
        elif feature == 'is_weekend':
            user_inputs['is_weekend'] = st.sidebar.checkbox(
                "Is Weekend?",
                value=(datetime.now().weekday() >= 5)
            )
        else:
            # Generic input for unknown features
            user_inputs[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                value=0.0
            )
    
    # Threshold settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Alert Settings")
    
    low_threshold = st.sidebar.slider(
        "Low Glucose Alert (mg/dL)",
        min_value=60, max_value=80, value=70,
        help="Alert when glucose falls below this level"
    )
    
    high_threshold = st.sidebar.slider(
        "High Glucose Alert (mg/dL)",
        min_value=160, max_value=200, value=180,
        help="Alert when glucose rises above this level"
    )
    
    # Prediction button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict Glucose", type="primary", use_container_width=True)
    
    # ============================================
    # MAIN CONTENT AREA
    # ============================================
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Trends", "ℹ️ Information"])
    
    with tab1:
        if predict_button:
            # Prepare input array in correct order
            input_array = np.array([user_inputs.get(feat, 0) for feat in feature_names])
            
            # Make prediction
            with st.spinner("Analyzing glucose patterns..."):
                predicted_glucose = predict_glucose(model, scaler_X, scaler_y, input_array, model_type)
            
            if predicted_glucose is not None:
                # Display prediction results
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Determine status and color
                    if predicted_glucose < low_threshold:
                        status = "⚠️ LOW"
                        color = "red"
                        alert_type = "error"
                    elif predicted_glucose > high_threshold:
                        status = "⚠️ HIGH"
                        color = "orange"
                        alert_type = "warning"
                    else:
                        status = "✅ NORMAL"
                        color = "green"
                        alert_type = "success"
                    
                    # Display predicted value
                    st.markdown(f"""
                        <div class='prediction-box'>
                            <h2>Predicted Glucose Level (30 min ahead)</h2>
                            <h1 style='color: {color}; font-size: 3.5rem; margin: 1rem 0;'>{predicted_glucose:.1f} mg/dL</h1>
                            <h3>{status}</h3>
                            <p style='color: gray; margin-top: 1rem;'>Model: {model_type} | RMSE: {metadata['performance']['RMSE']:.2f} mg/dL</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Display alerts
                st.markdown("---")
                
                if predicted_glucose < low_threshold:
                    st.error(f"""
                    ### ⚠️ HYPOGLYCEMIA ALERT
                    **Predicted glucose is below {low_threshold} mg/dL**
                    
                    **Immediate Actions Required:**
                    - 🍬 Consume 15-20g of fast-acting carbohydrates NOW
                    - 🥤 Drink 4 oz of fruit juice or regular soda
                    - 🍯 Take 3-4 glucose tablets or 1 tablespoon honey
                    - ⏰ Recheck glucose in 15 minutes
                    - 🔄 Repeat if still below {low_threshold} mg/dL
                    - 📞 Call emergency services if symptoms worsen
                    """)
                    
                elif predicted_glucose > high_threshold:
                    st.warning(f"""
                    ### ⚠️ HYPERGLYCEMIA ALERT
                    **Predicted glucose is above {high_threshold} mg/dL**
                    
                    **Recommended Actions:**
                    - 💉 Check if correction insulin is needed per your plan
                    - 💧 Drink water to stay hydrated
                    - 🚶 Consider light physical activity if safe
                    - 🥗 Avoid additional carbohydrates
                    - 📊 Check for ketones if glucose >250 mg/dL
                    - 📞 Contact healthcare provider if persistent
                    """)
                    
                else:
                    st.success(f"""
                    ### ✅ GLUCOSE IN TARGET RANGE
                    **Your predicted glucose is within the normal range ({low_threshold}-{high_threshold} mg/dL)**
                    
                    Continue with your current management plan and regular monitoring.
                    Next check recommended in 2-4 hours or as per your routine.
                    """)
                
                # Additional metrics
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    trend = "↑ Rising" if predicted_glucose > user_inputs.get('glucose_lag_1', 120) else "↓ Falling" if predicted_glucose < user_inputs.get('glucose_lag_1', 120) else "→ Stable"
                    st.metric("Trend", trend)
                
                with col2:
                    time_in_range = "Yes" if low_threshold <= predicted_glucose <= high_threshold else "No"
                    st.metric("In Target Range", time_in_range)
                
                with col3:
                    risk_level = "High" if predicted_glucose < 60 or predicted_glucose > 250 else "Moderate" if predicted_glucose < low_threshold or predicted_glucose > high_threshold else "Low"
                    st.metric("Risk Level", risk_level)
                
                with col4:
                    change = predicted_glucose - user_inputs.get('glucose_lag_1', 120)
                    st.metric("Change", f"{change:+.1f} mg/dL")
        
        else:
            # Instructions when no prediction made
            st.info("""
            ### 👋 Welcome to GlucoMind Dashboard
            
            **How to use:**
            1. Enter your recent glucose readings in the sidebar
            2. Add any insulin or carbohydrate information if available
            3. Adjust alert thresholds if needed
            4. Click **Predict Glucose** to see your 30-minute forecast
            
            The system uses a **{model}** model trained on glucose data to predict future values.
            """.format(model=model_type))
    
    with tab2:
        st.header("📈 Glucose Trends Visualization")
        
        # Generate sample historical data
        hours = pd.date_range(start=datetime.now() - timedelta(hours=24),
                              end=datetime.now(), freq='15min')
        
        # Create realistic glucose pattern
        base_glucose = 120
        glucose_history = []
        for i, hour in enumerate(hours):
            hour_of_day = hour.hour
            if 6 <= hour_of_day <= 8:
                modifier = 20
            elif 12 <= hour_of_day <= 14:
                modifier = 15
            elif 18 <= hour_of_day <= 20:
                modifier = 25
            elif 0 <= hour_of_day <= 6:
                modifier = -10
            else:
                modifier = 0
            
            glucose_val = base_glucose + modifier + np.random.normal(0, 10)
            glucose_history.append(glucose_val)
        
        # Create DataFrame
        df_history = pd.DataFrame({
            'Time': hours,
            'Glucose': glucose_history
        })
        
        # Add prediction point if available
        if 'predicted_glucose' in locals() and predicted_glucose is not None:
            future_time = datetime.now() + timedelta(minutes=30)
            df_prediction = pd.DataFrame({
                'Time': [datetime.now(), future_time],
                'Predicted': [glucose_history[-1], predicted_glucose]
            })
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add historical glucose trace
        fig.add_trace(go.Scatter(
            x=df_history['Time'],
            y=df_history['Glucose'],
            mode='lines',
            name='Historical Glucose',
            line=dict(color='blue', width=2),
            hovertemplate='Time: %{x}<br>Glucose: %{y:.1f} mg/dL<extra></extra>'
        ))
        
        # Add prediction if available
        if 'df_prediction' in locals():
            fig.add_trace(go.Scatter(
                x=df_prediction['Time'],
                y=df_prediction['Predicted'],
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=10),
                hovertemplate='Time: %{x}<br>Predicted: %{y:.1f} mg/dL<extra></extra>'
            ))
        
        # Add target range
        fig.add_hrect(
            y0=low_threshold, y1=high_threshold,
            fillcolor="green", opacity=0.1,
            annotation_text="Target Range",
            annotation_position="top left"
        )
        
        # Add threshold lines
        fig.add_hline(y=low_threshold, line_dash="dot", line_color="red",
                     annotation_text=f"Low: {low_threshold} mg/dL")
        fig.add_hline(y=high_threshold, line_dash="dot", line_color="orange",
                     annotation_text=f"High: {high_threshold} mg/dL")
        
        # Update layout
        fig.update_layout(
            title="24-Hour Glucose Trend with Prediction",
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            height=500,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 24-Hour Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Average', 'Std Dev', 'Minimum', 'Maximum', 'Time in Range'],
                'Value': [
                    f"{np.mean(glucose_history):.1f} mg/dL",
                    f"{np.std(glucose_history):.1f} mg/dL",
                    f"{np.min(glucose_history):.1f} mg/dL",
                    f"{np.max(glucose_history):.1f} mg/dL",
                    f"{np.sum((np.array(glucose_history) >= low_threshold) & (np.array(glucose_history) <= high_threshold)) / len(glucose_history) * 100:.1f}%"
                ]
            })
            st.table(stats_df)
        
        with col2:
            st.subheader("🎯 Glucose Distribution")
            hist_fig = px.histogram(
                glucose_history,
                nbins=30,
                title="Glucose Value Distribution",
                labels={'value': 'Glucose (mg/dL)', 'count': 'Frequency'}
            )
            hist_fig.add_vline(x=low_threshold, line_dash="dash", line_color="red")
            hist_fig.add_vline(x=high_threshold, line_dash="dash", line_color="orange")
            hist_fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(hist_fig, use_container_width=True)
    
    with tab3:
        st.header("ℹ️ Information & Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Understanding Your Results")
            st.markdown(f"""
            **Glucose Ranges:**
            - **Hypoglycemia:** < 70 mg/dL (Immediate action required)
            - **Normal:** 70-180 mg/dL (Target range)
            - **Hyperglycemia:** > 180 mg/dL (May require intervention)
            - **Severe Hyperglycemia:** > 250 mg/dL (Medical attention advised)
            
            **Current Model Performance:**
            - Model Type: **{model_type}**
            - R² Score: **{metadata['performance']['R2']:.4f}** ({metadata['performance']['R2']*100:.2f}% variance explained)
            - RMSE: **{metadata['performance']['RMSE']:.2f} mg/dL** (Average error)
            - MAE: **{metadata['performance']['MAE']:.2f} mg/dL**
            """)
            
            st.subheader("🔬 About the Model")
            model_descriptions = {
                'BiLSTM': "Bidirectional LSTM neural network that analyzes glucose patterns from both past and future directions",
                'LSTM': "Long Short-Term Memory neural network specialized in time-series prediction",
                'Linear Regression': "Traditional statistical model using linear relationships",
                'ARIMA': "AutoRegressive Integrated Moving Average time-series model"
            }
            
            st.markdown(f"""
            This dashboard uses a **{model_type}** model:
            {model_descriptions.get(model_type, "Advanced machine learning model")}
            
            **Key Features:**
            - Analyzes recent glucose trends
            - Accounts for time patterns
            - Provides 30-minute ahead predictions
            - Trained on real glucose data
            """)
        
        with col2:
            st.subheader("⚠️ Important Disclaimers")
            st.warning("""
            **Medical Disclaimer:**
            - This is an academic prototype, NOT a medical device
            - Do not make medical decisions based solely on these predictions
            - Always confirm with your glucose meter
            - Consult healthcare providers for medical advice
            - This tool is for educational and research purposes only
            """)
            
            st.subheader("📋 Best Practices")
            st.info("""
            **For Best Results:**
            1. Enter accurate recent glucose values
            2. Update insulin and carb data if available
            3. Use predictions as a supplement to regular monitoring
            4. Track prediction accuracy over time
            5. Report any consistent inaccuracies
            
            **When to Seek Medical Help:**
            - Glucose < 54 mg/dL (severe hypoglycemia)
            - Glucose > 300 mg/dL (risk of DKA)
            - Persistent symptoms despite normal readings
            - Any emergency symptoms
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        <p>GlucoMind v1.0 - Academic Research Project | University Final Year Project</p>
        <p>🚨 <strong>Not for Clinical Use</strong> - Consult Healthcare Professionals for Medical Decisions</p>
        <p>Model: {model_type} | RMSE: {metadata['performance']['RMSE']:.2f} mg/dL | R²: {metadata['performance']['R2']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()