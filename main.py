import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
import io
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Set page configuration
st.set_page_config(
    page_title="♻️ EcoClassify - Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for modern UI with better visibility
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
        
        .main {
            padding: 1rem 2rem;
            font-family: 'Poppins', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        .main-container {
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 3rem;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Mobile Responsive Design */
        @media (max-width: 768px) {
            .main {
                padding: 0.5rem 1rem;
            }
            
            .main-container {
                padding: 1.5rem;
                margin: 0.5rem;
                border-radius: 15px;
            }
            
            .hero-title {
                font-size: 2.5rem;
                margin-bottom: 0.3rem;
            }
            
            .hero-subtitle {
                font-size: 1.1rem;
                margin-bottom: 1.5rem;
            }
            
            .section-header {
                font-size: 1.3rem;
                margin: 1.5rem 0 1rem 0;
            }
            
            .result-card {
                padding: 1.5rem;
                margin: 1rem 0;
            }
            
            .result-card h3 {
                font-size: 1.5rem !important;
            }
            
            .info-card {
                padding: 1.5rem;
                margin: 1rem 0;
            }
            
            .upload-area {
                padding: 2rem 1rem;
                margin: 1.5rem 0;
                font-size: 1rem;
            }
            
            .stColumns > div {
                padding: 0 0.5rem;
            }
        }
        
        /* Tablet Responsive Design */
        @media (max-width: 1024px) and (min-width: 769px) {
            .main-container {
                padding: 2rem;
            }
            
            .hero-title {
                font-size: 3.2rem;
            }
            
            .hero-subtitle {
                font-size: 1.3rem;
            }
        }
        
        /* Small Mobile Design */
        @media (max-width: 480px) {
            .hero-title {
                font-size: 2rem;
            }
            
            .hero-subtitle {
                font-size: 1rem;
            }
            
            .main-container {
                padding: 1rem;
            }
            
            .result-card {
                padding: 1rem;
            }
            
            .info-card {
                padding: 1rem;
            }
            
            .section-header {
                font-size: 1.2rem;
            }
        }
        
        .hero-title {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            font-family: 'Poppins', sans-serif;
        }
        
        .hero-subtitle {
            font-size: 1.4rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
            font-family: 'Poppins', sans-serif;
        }
        
        .result-card {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: 0 15px 30px rgba(0, 184, 148, 0.4);
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .result-card h3 {
            font-size: 2rem !important;
            font-weight: 800 !important;
            margin-bottom: 1rem !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        
        .info-card {
            background: white;
            border: 2px solid #e8f4f8;
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
            border-color: #00b894;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .confidence-bar {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            overflow: hidden;
            height: 12px;
            margin: 1rem 0;
        }
        
        .confidence-fill {
            background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
            height: 100%;
            border-radius: 15px;
            transition: width 0.5s ease;
            box-shadow: 0 2px 10px rgba(0, 184, 148, 0.3);
        }
        
        .section-header {
            font-size: 1.6rem;
            font-weight: 700;
            color: #2c3e50;
            margin: 2rem 0 1.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            border-bottom: 3px solid #00b894;
            padding-bottom: 0.5rem;
            font-family: 'Poppins', sans-serif;
        }
        
        .sdg-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 0.5rem;
            margin: 1rem 0;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            background: rgba(102, 126, 234, 0.05);
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Optimized model loading with error handling
@st.cache_resource(show_spinner=False)
def load_model_optimized():
    """Load model with optimizations and comprehensive error handling"""
    try:
        # Configure TensorFlow for better performance
        tf.config.experimental.enable_memory_growth = True
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        # Load model with compile=False for faster loading
        model = tf.keras.models.load_model('keras_model.h5', compile=False)
        return model, "Model loaded successfully!"
    except Exception as e:
        error_msg = f"Model loading error: {str(e)}"
        return None, error_msg

# Optimized labels loading
@st.cache_data(show_spinner=False)
def load_labels():
    """Load and cache labels"""
    try:
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels, "Labels loaded successfully!"
    except FileNotFoundError:
        # Fallback labels if file not found
        return ["Biodegradable", "Non-Biodegradable"], "Using fallback labels"
    except Exception as e:
        return [], f"Error loading labels: {str(e)}"

# Optimized image preprocessing
def preprocess_image_fast(image):
    """Fast image preprocessing with optimizations"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize using fastest method for real-time processing
        image = ImageOps.fit(image, (224, 224), method=Image.Resampling.NEAREST)
        
        # Convert to numpy array and normalize
        img_array = np.asarray(image, dtype=np.float32)
        img_array = (img_array / 127.5) - 1  # Faster normalization
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    except Exception as e:
        return None, f"Image preprocessing error: {str(e)}"

# Improved async function for Ollama requests with better error handling
def get_environmental_info_async(label):
    """Asynchronous environmental info generation with improved connection handling"""
    try:
        clean_label = label.split(' ')[1].strip() if ' ' in label else label.strip()
        
        prompt = f"""
        Material: {clean_label}
        Provide information in this exact format (keep each section concise, 1-2 sentences max):

        Carbon Footprint: [X kg CO2 equivalent per kg]
        Environmental Impact: [Brief environmental effects]
        Disposal Method: [Best disposal/recycling method]
        Creative Ideas: [DIY projects or creative uses]
        Do you know?: [Interesting fact about this material]
        Reuse Ideas: [Creative reuse suggestions]
        """

        # Multiple request attempts with different configurations
        configs = [
            {"timeout": 45, "temperature": 0.7, "num_predict": 400},
            {"timeout": 60, "temperature": 0.5, "num_predict": 350},
            {"timeout": 75, "temperature": 0.3, "num_predict": 300}
        ]
        
        for i, config in enumerate(configs):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": config["temperature"],
                            "num_predict": config["num_predict"],
                            "top_k": 10,
                            "top_p": 0.9
                        }
                    },
                    timeout=config["timeout"]
                )

                if response.status_code == 200:
                    result = response.json().get("response", "")
                    if result.strip():
                        return result, None
                    
            except requests.exceptions.Timeout:
                if i < len(configs) - 1:
                    continue  # Try next configuration
                else:
                    return None, f"All requests timed out. Ollama may be processing a heavy load."
            except requests.exceptions.ConnectionError:
                return None, "Cannot connect to Ollama. Please check if it's running on localhost:11434"
        
        return None, "No valid response received from Ollama after multiple attempts."
                
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# SDG mapping for different waste types
SDG_MAPPING = {
    "cardboard": ["12", "13", "14", "15"],
    "glass": ["12", "14"],
    "metal": ["3", "6", "12", "14"],
    "paper": ["12", "13", "4", "15"],
    "plastic": ["6", "12", "14", "15"],
    "trash": ["3", "6", "12", "15"]
}

def main():
    # Header
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">♻️ EcoClassify</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Waste Classification for a Sustainable Future</p>', unsafe_allow_html=True)
    
    # Load model and labels
    model, model_status = load_model_optimized()
    labels, labels_status = load_labels()
    
    if model is None:
        st.error(f"⚠️ {model_status}")
        st.info("**Troubleshooting Steps:**")
        st.info("1. Install TensorFlow: `pip install tensorflow==2.10.0`")
        st.info("2. Ensure keras_model.h5 is in the current directory")
        st.info("3. Re-export model from Teachable Machine if needed")
        return
    
    # File uploader with custom styling
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📸 Upload an image of waste material",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">📷 Uploaded Image</div>', unsafe_allow_html=True)
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded image", use_container_width=True)
            
            # Process image
            with st.spinner("🔍 Analyzing image..."):
                img_array, preprocess_error = preprocess_image_fast(image)
                
                if preprocess_error:
                    st.error(f"Preprocessing error: {preprocess_error}")
                    return
                
                # Make prediction
                try:
                    prediction = model.predict(img_array, verbose=0)
                    predicted_index = np.argmax(prediction[0])
                    confidence = float(prediction[0][predicted_index])
                    predicted_label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    return
            
            # Display results
            st.markdown('<div class="section-header">🎯 Classification Results</div>', unsafe_allow_html=True)
            
            st.markdown(f'''
                <div class="result-card">
                    <h3 style="margin: 0;">🏷️ {predicted_label.upper()}</h3>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence * 100}%"></div>
                    </div>
                    <p style="margin: 0.5rem 0 0 0;">Confidence: {confidence:.1%}</p>
                </div>
            ''', unsafe_allow_html=True)
            
            # Display SDG goals if applicable
            waste_type = predicted_label.lower()
            sdg_goals = []
            for key, goals in SDG_MAPPING.items():
                if key in waste_type:
                    sdg_goals = goals
                    break
            
            if sdg_goals:
                st.markdown('<div class="section-header">🌍 Related UN SDG Goals</div>', unsafe_allow_html=True)
                sdg_cols = st.columns(len(sdg_goals))
                for i, goal in enumerate(sdg_goals):
                    with sdg_cols[i]:
                        try:
                            # Try multiple possible paths for SDG images
                            image_paths = [
                                f"sdg goals/{goal}.png",
                                f"sdg_goals/{goal}.png", 
                                f"SDG/{goal}.png",
                                f"images/sdg_{goal}.png",
                                f"assets/sdg{goal}.png"
                            ]
                            
                            image_loaded = False
                            for path in image_paths:
                                try:
                                    st.image(path, width=80)
                                    image_loaded = True
                                    break
                                except:
                                    continue
                            
                            if not image_loaded:
                                # Fallback display if no image found
                                st.markdown(f"""
                                <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                           border-radius: 10px; display: flex; align-items: center; justify-content: center; 
                                           color: white; font-weight: bold; font-size: 14px;">
                                    SDG {goal}
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.write(f"SDG {goal}")
        
        with col2:
            st.markdown('<div class="section-header">🌱 Environmental Impact Analysis</div>', unsafe_allow_html=True)
            
            # Environmental info container
            env_container = st.container()
            
            with env_container:
                # Use improved button styling and multiple connection attempts
                if st.button("🔍 Get Environmental Analysis", type="primary", use_container_width=True):
                    with st.spinner("🤖 Connecting to AI environmental expert..."):
                        # Create a progress indicator
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Progress updates
                        for i in range(3):
                            progress_bar.progress((i + 1) * 33)
                            status_text.text(f"Attempt {i + 1}/3: Connecting to Ollama...")
                            
                            # Get environmental info
                            env_info, env_error = get_environmental_info_async(predicted_label)
                            
                            if env_info:
                                break
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        
                        if env_error:
                            st.error(f"🚫 Connection Failed: {env_error}")
                            
                            with st.expander("🔧 Detailed Troubleshooting"):
                                st.markdown("""
                                **Check Ollama Status:**
                                ```bash
                                # Check if Ollama is running
                                curl http://localhost:11434/api/tags
                                
                                # Start Ollama if not running
                                ollama serve
                                
                                # In another terminal, run Mistral
                                ollama run mistral
                                ```
                                
                                **Common Solutions:**
                                - Restart Ollama service
                                - Check if port 11434 is blocked
                                - Verify Mistral model is downloaded
                                - Try running: `ollama pull mistral`
                                """)
                        else:
                            st.success("✅ Analysis Complete!")
                            
                            # Parse and display the response with better formatting
                            st.markdown('<div class="info-card">', unsafe_allow_html=True)
                            
                            # Split response into sections - Updated to include all sections
                            lines = env_info.split('\n')
                            sections = {}
                            current_section = None
                            
                            for line in lines:
                                line = line.strip()
                                # More flexible section detection
                                if ':' in line:
                                    # Check for section headers more broadly
                                    line_lower = line.lower()
                                    is_section = (
                                        'carbon footprint' in line_lower or 
                                        'environmental impact' in line_lower or 
                                        'disposal method' in line_lower or 
                                        'creative ideas' in line_lower or 
                                        'do you know' in line_lower or 
                                        'reuse ideas' in line_lower
                                    )
                                    
                                    if is_section:
                                        parts = line.split(':', 1)
                                        if len(parts) == 2:
                                            current_section = parts[0].strip()
                                            sections[current_section] = parts[1].strip()
                                elif current_section and line:
                                    sections[current_section] += f" {line}"
                            
                            # Display sections with improved styling and icons - Updated with new sections
                            section_icons = {
                                'Carbon Footprint': '🌡️',
                                'Environmental Impact': '🌍',
                                'Disposal Method': '♻️',
                                'Creative Ideas': '🎨',
                                'Do you know?': '🧠',
                                'Do you know': '🧠',  # Alternative format
                                'Reuse Ideas': '💡'
                            }
                            
                            # Order sections for better display
                            section_order = [
                                'Carbon Footprint', 'Environmental Impact', 'Disposal Method', 
                                'Creative Ideas', 'Do you know?', 'Do you know', 'Reuse Ideas'
                            ]
                            
                            # Display sections in order
                            displayed_sections = set()
                            for section_name in section_order:
                                for section, content in sections.items():
                                    if section.lower().strip() == section_name.lower().strip() and section not in displayed_sections:
                                        if content.strip():
                                            icon = section_icons.get(section_name, '📋')
                                            st.markdown(f"""
                                            <div style="margin: 1.5rem 0;">
                                                <h4 style="color: #2c3e50; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.8rem;">
                                                    {icon} {section}
                                                </h4>
                                                <p style="color: #34495e; font-size: 1.1rem; line-height: 1.7; margin-left: 2rem;">
                                                    {content}
                                                </p>
                                                <hr style="border: 1px solid #ecf0f1; margin: 1rem 0;">
                                            </div>
                                            """, unsafe_allow_html=True)
                                            displayed_sections.add(section)
                                            break
                            
                            # Display any remaining sections that weren't caught by the ordering
                            for section, content in sections.items():
                                if section not in displayed_sections and content.strip():
                                    icon = section_icons.get(section, '📋')
                                    st.markdown(f"""
                                    <div style="margin: 1.5rem 0;">
                                        <h4 style="color: #2c3e50; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.8rem;">
                                            {icon} {section}
                                        </h4>
                                        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.7; margin-left: 2rem;">
                                            {content}
                                        </p>
                                        <hr style="border: 1px solid #ecf0f1; margin: 1rem 0;">
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with app information
    st.markdown("---")
    with st.expander("ℹ️ About EcoClassify"):
        st.write("""
        **EcoClassify** is an AI-powered waste classification system that helps promote environmental awareness and sustainable practices.
        
        **Features:**
        - 🤖 AI-powered waste classification
        - 🌍 Environmental impact analysis
        - ♻️ Sustainable development goals mapping
        - 💡 Reuse and recycling suggestions
        
        **Supported Materials:** Cardboard, Glass, Metal, Paper, Plastic, General Trash
        """)
    
    with st.expander("🔧 Performance Tips"):
        st.write("""
        **For optimal performance:**
        - Use images smaller than 5MB
        - Ensure good lighting in photos
        - Keep Ollama running for environmental analysis
        - Close other resource-intensive applications
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()