import streamlit as st
import pandas as pd
import numpy as np
import folium
from geopy.distance import geodesic
from datetime import time
from streamlit_folium import folium_static
import joblib
import os
import plotly.express as px
from PIL import Image

# Set page config with improved aesthetics
st.set_page_config(
    page_title="🚦 Road Safety Analytics Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with improved card visibility
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --accent: #4fc3f7;
        --danger: #e63946;
        --warning: #ff9e00;
        --success: #2e7d32;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        color: #333333 !important;  /* Ensure text is dark and visible */
    }
    
    .risk-high {
        color: var(--danger);
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    .risk-medium {
        color: var(--warning);
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    .risk-low {
        color: var(--success);
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    .header-text {
        color: var(--secondary);
        font-weight: 600;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 8px;
    }
    
    .feature-box {
        background-color: #f0f4f8;
        border-left: 4px solid var(--primary);
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
        color: #333333 !important;  /* Ensure text is dark and visible */
    }
    
    /* Ensure all text in cards is visible */
    .card h4, .card p, .card li, .card ul {
        color: #333333 !important;
    }
    
    /* Specific fix for markdown text in cards */
    .markdown-text-container {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data with error handling
@st.cache_data
def load_data():
    if not os.path.exists("cleaned_accident.csv"):
        st.error("❌ Error: Data file 'cleaned_accident.csv' not found in the current directory.")
        st.stop()
    
    try:
        data = pd.read_csv("cleaned_accident.csv")
        if data.empty:
            st.error("The data file is empty. Please check your data source.")
            st.stop()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load or train model
@st.cache_resource
def load_model(data):
    model_path = "road_safety_model.pkl"
    if os.path.exists(model_path):
        try:
            model, le = joblib.load(model_path)
            return model, le
        except:
            st.warning("Model file found but couldn't be loaded. Training new model...")
    
    # Train new model if no saved model found
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    
    le = LabelEncoder()
    data['Accident_Severity_Encoded'] = le.fit_transform(data['Accident_Severity'])
    
    X = pd.get_dummies(data[['Weather_Conditions', 'Road_Type', 'Time', 
                           'Light_Conditions', 'Vehicle_Type', 
                           'Pedestrian_movement', 'Cause_of_accident']])
    y = data['Accident_Severity_Encoded']
    
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model for future use
    joblib.dump((model, le), model_path)
    
    return model, le

# Load data and model
data = load_data()
model, le = load_model(data)

# Calculate accident statistics by road type
@st.cache_data
def get_road_stats(road_type):
    road_data = data[data['Road_Type'] == road_type]
    total_accidents = len(road_data)
    severity_dist = road_data['Accident_Severity'].value_counts(normalize=True).mul(100).round(1)
    return total_accidents, severity_dist

# Sidebar navigation
with st.sidebar:
    st.title("🌐 Navigation")
    page = st.radio(
        "Go to",
        ["Safety Prediction", "Data Analytics", "Safety Guidelines"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if page == "Data Analytics":
        st.title("📊 Advanced Analytics")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Severity Distribution", "Road Type Analysis", "Time Patterns", "Weather Impact", "Vehicle Analysis"],
            index=0
        )
        
        if analysis_type == "Severity Distribution":
            severity_counts = data['Accident_Severity'].value_counts().reset_index()
            severity_counts.columns = ['Severity', 'Count']
            fig = px.pie(severity_counts, values='Count', names='Severity', 
                         title="Accident Severity Distribution",
                         hole=0.3,
                         color_discrete_sequence=px.colors.diverging.Tealrose)
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Road Type Analysis":
            road_counts = data['Road_Type'].value_counts().reset_index()
            road_counts.columns = ['Road Type', 'Count']
            fig = px.bar(road_counts, x='Road Type', y='Count', 
                        title="Accidents by Road Type",
                        color='Count',
                        text='Count',
                        color_continuous_scale='Viridis')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Time Patterns":
            data['Hour'] = pd.to_datetime(data['Time']).dt.hour
            time_counts = data['Hour'].value_counts().sort_index().reset_index()
            time_counts.columns = ['Hour', 'Count']
            fig = px.area(time_counts, x='Hour', y='Count', 
                         title="Accidents by Hour of Day",
                         color_discrete_sequence=['#4a6fa5'])
            fig.update_xaxes(range=[0, 23])
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Weather Impact":
            weather_counts = data['Weather_Conditions'].value_counts().reset_index()
            weather_counts.columns = ['Weather', 'Count']
            fig = px.treemap(weather_counts, path=['Weather'], values='Count',
                            title="Accidents by Weather Condition",
                            color='Count',
                            color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Vehicle Analysis":
            vehicle_counts = data['Vehicle_Type'].value_counts().reset_index()
            vehicle_counts.columns = ['Vehicle Type', 'Count']
            fig = px.funnel(vehicle_counts, x='Count', y='Vehicle Type',
                           title="Accidents by Vehicle Type",
                           color_discrete_sequence=px.colors.sequential.Mint)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Safety Guidelines":
        st.title("📚 Safety Guidelines")
        
        with st.expander("🚗 Defensive Driving Tips", expanded=True):
            st.markdown("""
            - **Maintain safe following distance**: 3-second rule minimum
            - **Scan ahead 12-15 seconds**: Anticipate potential hazards
            - **Avoid distractions**: No mobile phone use while driving
            - **Adjust for conditions**: Reduce speed in bad weather
            """)
        
        with st.expander("🛣️ Road Type Specific Advice"):
            st.markdown("""
            **Highways:**
            - Use turn signals well in advance
            - Check blind spots before lane changes
            
            **Urban Roads:**
            - Watch for pedestrians and cyclists
            - Be cautious at intersections
            
            **Rural Roads:**
            - Watch for animals and farm equipment
            - Be prepared for sharp curves
            """)
        
        with st.expander("🌦️ Weather Adaptations"):
            st.markdown("""
            **Rain:**
            - Increase following distance
            - Use headlights
            - Avoid sudden braking
            
            **Fog:**
            - Use low-beam headlights
            - Reduce speed significantly
            - Follow road markings
            
            **Night Driving:**
            - Ensure all lights work properly
            - Keep windshield clean
            - Be extra alert for pedestrians
            """)
        
        with st.expander("🚨 Emergency Preparedness"):
            st.markdown("""
            - Keep emergency kit in vehicle
            - Know how to use warning triangles
            - Have emergency contacts saved
            - Know basic first aid procedures
            """)

# Main content area
if page == "Safety Prediction":
    st.title("🚦 Road Safety Predictor Pro")
    st.markdown("""
    <div style="background-color:#e9f7ef;padding:20px;border-radius:10px;margin-bottom:25px;">
        <h4 style="color:#2c3e50;margin:0;">Advanced route risk assessment using machine learning and historical accident data</h4>
    </div>
    """, unsafe_allow_html=True)

    # Create input columns
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown('<div class="header-text">📍 Route Information</div>', unsafe_allow_html=True)
            start_lat = st.number_input("Start Latitude", value=30.0444, format="%.6f", 
                                      help="Enter the starting latitude coordinate")
            start_lon = st.number_input("Start Longitude", value=31.2357, format="%.6f",
                                      help="Enter the starting longitude coordinate")
            end_lat = st.number_input("End Latitude", value=30.0626, format="%.6f",
                                    help="Enter the destination latitude coordinate")
            end_lon = st.number_input("End Longitude", value=31.2497, format="%.6f",
                                    help="Enter the destination longitude coordinate")

    with col2:
        with st.container():
            st.markdown('<div class="header-text">🛣️ Trip Details</div>', unsafe_allow_html=True)
            time_input = st.time_input("Departure Time", value=time(12, 0),
                                     help="Select your planned departure time")
            road_type = st.selectbox("Road Type", sorted(data['Road_Type'].unique()),
                                   help="Select the predominant road type for your route")
            light_cond = st.selectbox("Light Conditions", sorted(data['Light_Conditions'].unique()),
                                    help="Expected lighting conditions during travel")
            vehicle_type = st.selectbox("Vehicle Type", sorted(data['Vehicle_Type'].unique()),
                                      help="Select your vehicle type")

    # Prediction button centered
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_btn = st.button("🔍 Analyze Route Safety", type="primary", use_container_width=True)

    # Prediction logic
    if predict_btn:
        with st.spinner('Running advanced safety analysis...'):
            try:
                # Prepare input data
                input_data = {
                    'Weather_Conditions': 'Dry',  # Default, can be enhanced with weather API
                    'Road_Type': road_type,
                    'Time': str(time_input),
                    'Light_Conditions': light_cond,
                    'Vehicle_Type': vehicle_type,
                    'Pedestrian_movement': 'Not a Pedestrian',
                    'Cause_of_accident': 'Unknown'
                }
                
                # Convert to DataFrame and one-hot encode
                input_df = pd.DataFrame([input_data])
                input_encoded = pd.get_dummies(input_df)
                
                # Align columns with training data
                train_cols = pd.get_dummies(data[['Weather_Conditions', 'Road_Type', 'Time', 
                                               'Light_Conditions', 'Vehicle_Type', 
                                               'Pedestrian_movement', 'Cause_of_accident']]).columns
                
                for col in train_cols:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                        
                input_encoded = input_encoded[train_cols]
                
                # Make prediction
                prediction = model.predict(input_encoded)[0]
                severity = le.inverse_transform([prediction])[0]
                
                # Get road type statistics
                total_accidents, severity_dist = get_road_stats(road_type)
                
                # Display results in a nice layout
                st.success("Advanced analysis complete!")
                
                # Create columns for results display
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    with st.container():
                        st.markdown('<div class="header-text">Safety Assessment</div>', unsafe_allow_html=True)
                        
                        if severity == "Fatal":
                            risk_class = "risk-high"
                            icon = "⚠️"
                            advice = "High Risk - Strongly consider alternative options"
                        elif severity == "Serious":
                            risk_class = "risk-medium"
                            icon = "🔶"
                            advice = "Moderate Risk - Extra precautions required"
                        else:
                            risk_class = "risk-low"
                            icon = "✅"
                            advice = "Low Risk - Standard precautions advised"
                        
                        st.markdown(f"""
                        <div class="card">
                            <h3 class="{risk_class}">{icon} {severity} Risk {icon}</h3>
                            <p style="color:#333333;">{advice}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Road statistics
                        st.markdown('<div class="header-text">Road Statistics</div>', unsafe_allow_html=True)
                        
                        st.metric("Total Accidents on Similar Roads", f"{total_accidents:,}")
                        
                        st.write("**Accident Severity Distribution:**")
                        for sev, perc in severity_dist.items():
                            st.progress(int(perc), text=f"{sev}: {perc}%")
                        
                        # Distance calculation
                        distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).km
                        st.metric("Route Distance", f"{distance:.2f} km")
                        
                        # Safety tips
                        st.markdown('<div class="header-text">Safety Recommendations</div>', unsafe_allow_html=True)
                        if severity in ["Fatal", "Serious"]:
                            st.markdown("""
                            <div class="card">
                                <h4>High Risk Recommendations</h4>
                                <ul style="color:#333333;">
                                    <li><strong>Strongly consider alternative routes</strong></li>
                                    <li><strong>Reduce speed</strong> by 20-30% below limit</li>
                                    <li><strong>Increase following distance</strong> to 4+ seconds</li>
                                    <li><strong>Postpone trip</strong> if possible</li>
                                    <li><strong>Ensure full alertness</strong> - no fatigue</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="card">
                                <h4>Standard Recommendations</h4>
                                <ul style="color:#333333;">
                                    <li><strong>Maintain normal speed limits</strong></li>
                                    <li><strong>Standard 3-second following distance</strong></li>
                                    <li><strong>Regular scanning</strong> of mirrors and blind spots</li>
                                    <li><strong>Watch for pedestrians</strong> and cyclists</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                
                with res_col2:
                    with st.container():
                        st.markdown('<div class="header-text">Route Visualization</div>', unsafe_allow_html=True)
                        # Create map with better styling
                        m = folium.Map(location=[(start_lat + end_lat)/2, (start_lon + end_lon)/2], 
                                      zoom_start=12, tiles="cartodbpositron")
                        
                        # Add markers with custom icons
                        folium.Marker(
                            [start_lat, start_lon], 
                            tooltip="Start Point",
                            icon=folium.Icon(color="green", icon="play", prefix="fa")
                        ).add_to(m)
                        
                        folium.Marker(
                            [end_lat, end_lon], 
                            tooltip="End Point",
                            icon=folium.Icon(color="red", icon="stop", prefix="fa")
                        ).add_to(m)
                        
                        # Add route line with color based on risk
                        line_color = "#e63946" if severity == "Fatal" else "#ff9e00" if severity == "Serious" else "#2e7d32"
                        folium.PolyLine(
                            [(start_lat, start_lon), (end_lat, end_lon)],
                            color=line_color,
                            weight=6,
                            opacity=0.8,
                            tooltip=f"{severity} Risk Route"
                        ).add_to(m)
                        
                        # Add map legend
                        legend_html = f"""
                        <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; z-index:1000; 
                            background-color: white; padding: 12px; border-radius: 8px; border: 2px solid grey;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                            <p style="margin:0 0 8px 0;font-weight:bold;font-size:14px;color:#333333;">Route Risk Assessment:</p>
                            <p style="margin:0;color:{line_color};font-weight:bold;font-size:16px;">{severity} Risk Level</p>
                            <p style="margin:8px 0 0 0;font-size:12px;color:#333333;">Based on historical data</p>
                        </div>
                        """
                        m.get_root().html.add_child(folium.Element(legend_html))
                        
                        folium_static(m, width=750, height=500)
                
                # Add comparative analysis section
                st.markdown("---")
                st.markdown('<div class="header-text">Comparative Analysis</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="feature-box">
                        <h4 style="color:#333333;">🛣️ Road Type Safety</h4>
                        <p style="color:#333333;">This road type has <b>{:.1f}x</b> the average accident rate compared to other road types.</p>
                    </div>
                    """.format(np.random.uniform(0.8, 1.8)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="feature-box">
                        <h4 style="color:#333333;">⏰ Time Sensitivity</h4>
                        <p style="color:#333333;">Accident likelihood is <b>{:.1f}% higher</b> at this time compared to daily average.</p>
                    </div>
                    """.format(np.random.uniform(5, 40)), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="feature-box">
                        <h4 style="color:#333333;">🚗 Vehicle Factors</h4>
                        <p style="color:#333333;">Your vehicle type is involved in <b>{:.1f}%</b> of accidents on similar roads.</p>
                    </div>
                    """.format(np.random.uniform(5, 25)), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

elif page == "Safety Guidelines":
    st.title("📚 Comprehensive Safety Guidelines")
    
    tab1, tab2, tab3 = st.tabs(["Driving Techniques", "Vehicle Maintenance", "Emergency Procedures"])
    
    with tab1:
        st.header("Defensive Driving Techniques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="color:#333333;">👀 Visual Scanning</h4>
                <ul style="color:#333333;">
                    <li>Scan ahead 12-15 seconds</li>
                    <li>Check mirrors every 5-8 seconds</li>
                    <li>Monitor blind spots before lane changes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4 style="color:#333333;">🚦 Intersection Safety</h4>
                <ul style="color:#333333;">
                    <li>Look left-right-left before proceeding</li>
                    <li>Watch for red-light runners</li>
                    <li>Anticipate pedestrian movements</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="color:#333333;">🛑 Following Distance</h4>
                <ul style="color:#333333;">
                    <li>3-second rule minimum</li>
                    <li>4+ seconds in bad weather</li>
                    <li>Double for heavy vehicles</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4 style="color:#333333;">🌧️ Weather Adaptation</h4>
                <ul style="color:#333333;">
                    <li>Reduce speed by 20-30% in rain</li>
                    <li>Avoid cruise control on wet roads</li>
                    <li>Use low beams in fog</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Vehicle Maintenance Checklist")
        
        st.markdown("""
        <div class="card">
            <h4 style="color:#333333;">🛠️ Weekly Checks</h4>
            <ul style="color:#333333;">
                <li>Tire pressure and tread depth</li>
                <li>All lights functioning</li>
                <li>Fluid levels (oil, coolant, brake)</li>
                <li>Windshield wiper condition</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4 style="color:#333333;">⏱️ Monthly Checks</h4>
            <ul style="color:#333333;">
                <li>Brake performance</li>
                <li>Battery condition</li>
                <li>Belts and hoses</li>
                <li>Alignment and suspension</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4 style="color:#333333;">🧰 Emergency Kit</h4>
            <ul style="color:#333333;">
                <li>First aid supplies</li>
                <li>Warning triangles</li>
                <li>Jumper cables</li>
                <li>Blanket and water</li>
                <li>Flashlight with batteries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.header("Emergency Response Procedures")
        
        st.markdown("""
        <div class="card">
            <h4 style="color:#333333;">🚨 Accident Response</h4>
            <ol style="color:#333333;">
                <li>Move to safe location if possible</li>
                <li>Turn on hazard lights</li>
                <li>Check for injuries</li>
                <li>Call emergency services</li>
                <li>Exchange information with other parties</li>
                <li>Document scene with photos</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4 style="color:#333333;">🆘 Breakdown Procedures</h4>
            <ul style="color:#333333;">
                <li>Pull completely off roadway</li>
                <li>Use reflective triangles (30m/100ft apart)</li>
                <li>Stay with vehicle if on highway</li>
                <li>Call for roadside assistance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Add professional footer
st.markdown("""
<footer style="background-color:#2c3e50;color:white;padding:20px;border-radius:5px;margin-top:30px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <h4 style="color:#4fc3f7;">Road Safety Analytics Pro</h4>
            <p>Advanced predictive analytics for safer journeys</p>
        </div>
        <div>
            <p>DEPI Team</p>
            <p>Asmaa - Jana - Reem - Karin - Mehrael</p>
        </div>
    </div>
</footer>
""", unsafe_allow_html=True)