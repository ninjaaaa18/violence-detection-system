import cv2
import numpy as np
import streamlit as st
from pillow_heif import register_heif_opener
from PIL import Image
import time
from datetime import datetime

# Register HEIC format support
register_heif_opener()

# Page configuration
st.set_page_config(
    page_title="SentinelAI - Violence Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and professional styling
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0e27;
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f1535;
        border-right: 2px solid #1a1f3a;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    .main {
        background-color: #0a0e27;
    }
    
    [data-testid="stMetric"] {
        background-color: #1a1f3a;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a3050;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1f3a;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0a0a0;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ff1744;
        border-bottom: 3px solid #ff1744;
    }
    
    .status-banner {
        background: linear-gradient(135deg, #1a1f3a 0%, #2a3050 100%);
        border-radius: 16px;
        padding: 24px;
        border: 2px solid #ff1744;
        margin-bottom: 24px;
        text-align: center;
    }
    
    .status-safe {
        border-color: #00c853 !important;
    }
    
    .status-text-alert {
        font-size: 32px;
        font-weight: bold;
        color: #ff1744;
        text-shadow: 0 0 10px rgba(255, 23, 68, 0.5);
        margin-bottom: 10px;
    }
    
    .status-text-safe {
        font-size: 32px;
        font-weight: bold;
        color: #00c853;
        text-shadow: 0 0 10px rgba(0, 200, 83, 0.5);
        margin-bottom: 10px;
    }
    
    .confidence-score {
        font-size: 48px;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .confidence-alert {
        color: #ff1744;
    }
    
    .confidence-safe {
        color: #00c853;
    }
    
    .subtitle {
        color: #a0a0a0;
        font-size: 14px;
        margin-top: 8px;
    }
    
    .sidebar-title {
        color: #ff1744;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .card {
        background-color: #1a1f3a;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a3050;
        margin-bottom: 16px;
    }
    
    .video-container {
        background-color: #0f1535;
        border-radius: 12px;
        border: 2px solid #2a3050;
        padding: 0;
        overflow: hidden;
    }
    
    .detection-result {
        background: linear-gradient(135deg, #1a1f3a 0%, #2a3050 100%);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #2a3050;
        margin-top: 16px;
        text-align: center;
    }
    
    .result-label {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 12px;
    }
    
    .result-alert {
        color: #ff1744;
    }
    
    .result-safe {
        color: #00c853;
    }
    
    .timestamp {
        color: #707070;
        font-size: 12px;
        margin-top: 12px;
    }
    
    .history-item {
        background-color: #0f1535;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #2a3050;
        margin-bottom: 10px;
        font-size: 13px;
    }
    
    .history-item-alert {
        border-left-color: #ff1744;
    }
    
    .history-item-safe {
        border-left-color: #00c853;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #1a1f3a 0%, #2a3050 100%);
        border-radius: 12px;
        border: 2px dashed #2a3050;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    
    button {
        background-color: #ff1744 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
    
    button:hover {
        background-color: #e0143a !important;
    }
    
    .stCheckbox {
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor_model():
    from model import Model
    model = Model()
    return model

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'current_detection' not in st.session_state:
    st.session_state.current_detection = None

model = get_predictor_model()

# Sidebar Navigation
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🛡️ SentinelAI</div>", unsafe_allow_html=True)
    st.markdown("### Violence Detection System")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigation",
        ["📊 Dashboard", "🎥 Live Detection", "📤 Image Upload"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    st.markdown(f"**Status:** 🟢 Active", unsafe_allow_html=True)
    st.markdown("**Model:** MobileNetV2", unsafe_allow_html=True)
    st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}", unsafe_allow_html=True)

# Main content area
if nav_option == "📊 Dashboard":
    # Top status banner
    st.markdown("""
    <div class='card'>
        <h2 style='color: #a0a0a0; margin-bottom: 20px;'>Welcome to SentinelAI</h2>
        <p style='color: #707070; line-height: 1.6;'>
            Real-time violence detection system powered by deep learning. 
            Monitor live feeds and analyze images for potential violent activities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick access cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <div style='font-size: 24px; margin-bottom: 10px;'>🎥</div>
            <div style='color: #a0a0a0; font-size: 14px;'>Live Detection</div>
            <div style='color: #ff1744; font-size: 20px; font-weight: bold; margin-top: 10px;'>Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <div style='font-size: 24px; margin-bottom: 10px;'>📤</div>
            <div style='color: #a0a0a0; font-size: 14px;'>Image Upload</div>
            <div style='color: #00c853; font-size: 20px; font-weight: bold; margin-top: 10px;'>Active</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <div style='font-size: 24px; margin-bottom: 10px;'>📋</div>
            <div style='color: #a0a0a0; font-size: 14px;'>Detection History</div>
            <div style='color: #a0a0a0; font-size: 20px; font-weight: bold; margin-top: 10px;'>""" + str(len(st.session_state.detection_history)) + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detection history section
    if st.session_state.detection_history:
        st.markdown("### Recent Detections")
        history_col1, history_col2 = st.columns(2)
        with history_col1:
            st.markdown("<div class='card'><h4>Latest Detections</h4>", unsafe_allow_html=True)
            for item in st.session_state.detection_history[-5:]:
                alert_class = "history-item-alert" if "violence" in item['label'].lower() else "history-item-safe"
                icon = "⚠️" if "violence" in item['label'].lower() else "✅"
                st.markdown(f"""
                <div class='history-item {alert_class}'>
                    {icon} <strong>{item['label']}</strong> - {item['confidence']:.1%}<br>
                    <span style='color: #707070; font-size: 11px;'>{item['timestamp']}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif nav_option == "🎥 Live Detection":
    st.markdown("### 📹 Real-Time Webcam Detection", unsafe_allow_html=True)
    
    # Control row
    col_ctrl1, col_ctrl2 = st.columns([5, 1])
    with col_ctrl2:
        enable_webcam = st.checkbox("Enable", value=False)
    
    # Main layout: Video feed on left, Status on right
    left_col, right_col = st.columns([0.62, 0.38], gap="medium")
    
    with left_col:
        st.markdown("#### 🎥 Live Feed", unsafe_allow_html=True)
        video_placeholder = st.empty()
    
    with right_col:
        st.markdown("#### ⚡ Status", unsafe_allow_html=True)
        status_placeholder = st.empty()
        
        st.markdown("---")
        st.markdown("#### 📊 Metrics", unsafe_allow_html=True)
        metric_col1, metric_col2 = st.columns(2)
        metric_label_placeholder = metric_col1.empty()
        metric_conf_placeholder = metric_col2.empty()
        
        st.markdown("---")
        st.markdown("#### 🖥️ System Info", unsafe_allow_html=True)
        system_placeholder = st.empty()
    
    if enable_webcam:
        cap = cv2.VideoCapture(0)
        
        frame_count = 0
        latest_confidence = 0
        latest_label = "Initializing..."
        latest_is_violence = False
        
        try:
            while cap.isOpened() and enable_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame. Please check your webcam.")
                    break
                
                # Resize for display (reduced height to ~420px)
                display_frame = cv2.resize(frame, (560, 420))
                
                # Predict every 5 frames
                if frame_count % 5 == 0:
                    result = model.predict(image=frame)
                    latest_label = result["label"].title()
                    latest_confidence = result.get("confidence", 0)
                    latest_is_violence = "violence" in latest_label.lower()
                    
                    # Add to history
                    st.session_state.detection_history.append({
                        'label': latest_label,
                        'confidence': latest_confidence,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
                    if len(st.session_state.detection_history) > 50:
                        st.session_state.detection_history.pop(0)
                
                # Display video
                video_placeholder.image(display_frame, channels="BGR", use_container_width=True)
                
                # Display status banner
                if latest_is_violence:
                    status_placeholder.markdown(f"""
                    <div class='status-banner' style='margin-bottom: 0;'>
                        <div class='status-text-alert'>⚠️ VIOLENCE</div>
                        <div class='status-text-alert' style='font-size: 18px; text-shadow: none; margin-top: 5px;'>DETECTED</div>
                        <div class='confidence-score confidence-alert' style='margin: 12px 0;'>{latest_confidence:.1%}</div>
                        <div class='subtitle' style='color: #ff9999;'>Immediate attention required</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status_placeholder.markdown(f"""
                    <div class='status-banner status-safe' style='margin-bottom: 0;'>
                        <div class='status-text-safe'>✅ SAFE</div>
                        <div class='confidence-score confidence-safe' style='margin: 12px 0;'>{latest_confidence:.1%}</div>
                        <div class='subtitle' style='color: #99ff99;'>Normal activity</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Update metrics
                metric_label_placeholder.metric("Classification", latest_label, label_visibility="hidden")
                metric_conf_placeholder.metric("Confidence", f"{latest_confidence:.1%}", label_visibility="hidden")
                
                # System info
                system_placeholder.markdown(f"""
                <div class='card' style='margin-bottom: 0;'>
                    <div style='font-size: 13px; color: #a0a0a0;'>
                        <div>🔢 Frames: <strong>{frame_count}</strong></div>
                        <div style='margin-top: 8px;'>⏱️ Time: <strong>{datetime.now().strftime('%H:%M:%S')}</strong></div>
                        <div style='margin-top: 8px;'>📡 FPS: <strong>~33</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                frame_count += 1
                time.sleep(0.03)
        
        except Exception as e:
            st.error(f"Error during detection: {e}")
        finally:
            cap.release()

elif nav_option == "📤 Image Upload":
    st.markdown("### 📸 Upload Image for Analysis", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select an image file",
        type=["heic", "jpg", "jpeg", "png"],
        help="Supported formats: HEIC, JPG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Uploaded Image", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("#### Analysis Results", unsafe_allow_html=True)
                
                # Perform prediction
                image_np = np.array(image)
                result = model.predict(image=image_np)
                
                label = result['label'].title()
                confidence = result.get('confidence', 0)
                is_violence = "violence" in label.lower()
                
                # Add to history
                st.session_state.detection_history.append({
                    'label': label,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                if len(st.session_state.detection_history) > 50:
                    st.session_state.detection_history.pop(0)
                
                # Display status banner
                if is_violence:
                    st.markdown(f"""
                    <div class='status-banner'>
                        <div class='status-text-alert'>⚠️ VIOLENCE DETECTED</div>
                        <div class='confidence-score confidence-alert'>{confidence:.1%}</div>
                        <div class='subtitle'>Take appropriate action</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='status-banner status-safe'>
                        <div class='status-text-safe'>✅ SAFE</div>
                        <div class='confidence-score confidence-safe'>{confidence:.1%}</div>
                        <div class='subtitle'>No violence detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Classification", label)
                with col_m2:
                    st.metric("Confidence", f"{confidence:.1%}")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")