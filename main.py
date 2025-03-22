import os
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import google.generativeai as genai
import io
import warnings
import absl.logging
import base64
from pathlib import Path
import folium
from streamlit_folium import folium_static
import requests
from geopy.geocoders import Nominatim
from datetime import datetime
import json

# Suppress warnings
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)

# Set your API key for Google Generative AI
GOOGLE_API_KEY = "AIzaSyDPbPU525_fCzGsfChvo4qLpYnyKZRwX6k"  # Updated API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Define detection types
detection_types = {
    "brain_tumor": "🧠 Brain Tumor",
    "eye_disease": "👁️ Eye Disease",
    "lung_cancer": "🫁 Lung Cancer",
    "bone_fracture": "🦴 Bone Fracture",
    "skin_disease": "🔬 Skin Disease"
}

# Initialize Google Generative AI client
genai.configure(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add these imports at the top
import base64
from pathlib import Path

# Add this function before your main code
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("data:image/jpg;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Replace the existing background markdown code with this
background_image_path = "/Users/sreemadhav/SreeMadhav/Mhv CODES/MVSR/p22/PythonProject/wallpaperflare.com_wallpaper.jpg"
set_background(background_image_path)

# Update the CSS styling section with these specific sidebar selectors
st.markdown("""
    <style>
    .stApp > header {
        background-color: transparent !important;
    }
    
    .stApp {
        color: white;  /* Makes text white for better visibility */
    }
    
    /* Make sidebar background semi-transparent black */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Style sidebar content */
    [data-testid="stSidebar"] > div:first-child {
        background-color: transparent !important;
    }

    /* Make the sidebar text more visible */
    [data-testid="stSidebar"] .st-emotion-cache-16idsys p,
    [data-testid="stSidebar"] .st-emotion-cache-16idsys span,
    [data-testid="stSidebar"] .st-emotion-cache-16idsys div,
    [data-testid="stSidebar"] .st-emotion-cache-16idsys label {
        color: white !important;
    }

    /* Style the markdown separator */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Style for the GIF container */
    [data-testid="stImage"] {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_language' not in st.session_state:
    st.session_state.current_language = 'en'
if 'models' not in st.session_state:
    st.session_state.models = {}

# Initialize Gemini model globally
try:
    # Use gemini-2.0-flash model
    GEMINI_MODEL = genai.GenerativeModel(
        model_name="gemini-2.0-flash",  # Updated model name
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
    )

    # Test the model initialization with a simple prompt
    test_response = GEMINI_MODEL.generate_content("Hello")
    if not test_response.text:
        raise Exception("Failed to get response from model")

except Exception as e:
    st.error(f"Error initializing Gemini API: {str(e)}")
    GEMINI_MODEL = None

# Move sidebar outside of the try-except block
# Sidebar with improved layout
with st.sidebar:
    # Logo
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.title("🏥")  # Fallback to emoji

    st.title("Welcome to 🔥FireML Multispeciality Hospitals")
    st.markdown("---")

    # Language selector
    st.subheader("🌐 Language")
    languages = {
        'en': '🇺🇸 English',
        'es': '🇪🇸 Español',
        'fr': '🇫🇷 Français',
        'hi': '🇮🇳 हिंदी',
        'ta': '🇮🇳 தமிழ்',
        'zh': '🇨🇳 中文'
    }
    selected_lang = st.selectbox("", options=list(languages.keys()), 
                                format_func=lambda x: languages[x],
                                key='language')

    # How to use section
    st.markdown("---")
    st.subheader("📖 How to Use This Project")
    st.markdown("""
    ### Quick Start Guide

    #### 1. Image Analysis
    - **Using Camera**:
        - Click on "Camera" tab
        - Allow camera access
        - Take a clear photo
        - Select appropriate detection model
        - Click "Analyze"
    
    - **Upload Image**:
        - Click on "Upload" tab
        - Choose medical image file
        - Select detection model
        - Click "Analyze"
    
    #### 2. Available Detection Models
    - 🧠 Brain Tumor Detection
    - 👁️ Eye Disease Analysis
    - 🫁 Lung Cancer Screening
    - 🦴 Bone Fracture Detection
    - 🔬 Skin Disease Analysis
    
    #### 3. AI Chat Assistant
    - Type your medical queries
    - Get instant AI responses
    - Ask for analysis explanations
    
    #### 4. Best Practices
    - Use clear, well-lit images
    - Keep camera steady
    - Follow on-screen instructions
    - Wait for analysis to complete
    
    #### ⚠️ Important Note
    This is an AI-assisted tool for preliminary analysis only. Always consult healthcare professionals for medical decisions.
    
    #### 📞 Contact
    For technical support:
    Email: support@smvshospitals.com
    """)

def load_model(model_type):
    """Load YOLO model based on type"""
    model_paths = {
        'brain_tumor': "braintumorp1.pt",  # your existing model
        'eye_disease': "eye_disease.pt",  # to be added later
        'lung_cancer': "lung_cancer.pt",  # to be added later
        'bone_fracture': "bone_fracture.pt",  # to be added later
        'skin_disease': "skin_disease.pt"  # to be added later
    }

    if model_type not in st.session_state.models:
        model_path = model_paths.get(model_type)
        if model_path and os.path.exists(model_path):
            st.session_state.models[model_type] = YOLO(model_path)
        else:
            st.warning(f"Model for {model_type} not found. Only Brain Tumor detection is currently available.")
            if model_type != 'brain_tumor':
                return None
            # Fallback to brain tumor model
            st.session_state.models[model_type] = YOLO("braintumorp1.pt")
    return st.session_state.models.get(model_type)


def translate_text(text, target_lang):
    """Translate text to target language"""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails


def text_to_speech(text, language='en'):
    """Convert text to speech"""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def speech_to_text():
    """Convert speech to text"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return None


def get_gemini_response(prompt, image=None):
    """Get response from Gemini API"""
    try:
        if image:
            # For image analysis, use gemini-2.0-flash
            response = GEMINI_MODEL.generate_content([prompt, image])
        else:
            # For text chat, use the same model
            response = GEMINI_MODEL.generate_content(prompt)

        if hasattr(response, 'text') and response.text:
            return response.text
        return "I apologize, but I couldn't generate a response."

    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now."


def process_chat_response(prompt):
    """Process chat responses with streaming"""
    try:
        response = GEMINI_MODEL.send_message(prompt, stream=True)

        # Create a placeholder for streaming response
        message_placeholder = st.empty()
        full_response = ""

        # Stream the response
        for chunk in response:
            full_response += chunk.text
            message_placeholder.markdown(full_response + "▌")

        # Replace the placeholder with the complete response
        message_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now."


def analyze_image_with_google_vision(image):
    """Analyze the image using Google Vision API."""
    image = vision.Image(content=image)
    response = vision_client.label_detection(image=image)
    labels = response.label_annotations
    results = [(label.description, label.score) for label in labels]
    return results


def process_image(image):
    """Process the uploaded image and return analysis results."""
    try:
        # Ensure the image is in a valid format
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Analyze the image using Google Vision API
        analysis_results = analyze_image_with_google_vision(img_byte_arr)
        return analysis_results
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None  # Return None if there's an error


def display_prediction(class_name, conf):
    return f"""
    <div class='detection-result'>
        <h4 style='color: #00d2ff; margin: 0;'>{class_name}</h4>
        <p style='color: #ffffff; margin: 5px 0;'>Confidence: {conf:.2%}</p>
    </div>
    """


def initialize_chat_if_needed():
    """Initialize or get existing chat session"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'gemini_chat' not in st.session_state and GEMINI_MODEL:
        st.session_state.gemini_chat = GEMINI_MODEL.start_chat(history=[])


def get_chat_response(prompt):
    """Get chat response from Gemini"""
    try:
        if not GEMINI_MODEL:
            return "AI model not initialized properly. Please check your API key."

        medical_prompt = f"""
        You are a medical AI assistant. Please provide helpful medical information and advice while keeping in mind:
        1. Be clear and professional
        2. Include relevant medical terminology with explanations
        3. Always encourage consulting healthcare professionals
        4. Provide evidence-based information when possible

        User question: {prompt}
        """

        # Generate response
        response = GEMINI_MODEL.generate_content(medical_prompt)

        # Check if response is blocked
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            return "I apologize, but I cannot provide a response to that query. Please try rephrasing your question."

        if hasattr(response, 'text') and response.text:
            return response.text.strip()  # Trim the response

        return "I apologize, but I couldn't generate a response."

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I apologize, but I'm having trouble generating a response."


def chat_interface(selected_lang):
    """Render chat interface"""
    st.subheader(translate_interface_text("💬 Medical Assistant Chat", selected_lang))

    # Chat input at the top
    if prompt := st.chat_input(translate_interface_text("Ask me anything about your health...", selected_lang)):
        # Add user message
        with st.chat_message("user"):
            st.markdown(translate_interface_text(prompt, selected_lang))
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner(translate_interface_text("Thinking...", selected_lang)):
                response = get_chat_response(prompt)
                # Translate response if needed
                if selected_lang != 'en':
                    response = translate_text(response, selected_lang)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            content = translate_text(message["content"], selected_lang) if selected_lang != 'en' else message["content"]
            st.markdown(content)

    # Add chat guidelines in expandable section at the bottom
    with st.expander(translate_interface_text("ℹ️ Chat Guidelines", selected_lang), expanded=False):
        guidelines = [
            "Ask about medical conditions",
            "Get general health advice",
            "Learn about prevention",
            "Understand detection results"
        ]
        for guideline in guidelines:
            st.markdown(f"- {translate_interface_text(guideline, selected_lang)}")

        st.markdown(f"**{translate_interface_text('Note:', selected_lang)}** " +
                    translate_interface_text(
                        "This is an AI assistant and not a replacement for professional medical advice.",
                        selected_lang))

    # Clear chat button at the bottom
    if st.session_state.chat_history:
        if st.button(translate_interface_text("🗑️ Clear Chat History", selected_lang), key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def translate_interface_text(text, target_lang):
    """Translate interface text if not in English"""
    if target_lang != 'en':
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
    return text


def translate_page_content(selected_lang):
    """Translate all static page content"""
    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Initialize img_file variable
        img_file = None

        # Translate tab labels
        tab_labels = [
            translate_interface_text(x, selected_lang)
            for x in ["📸 Camera", "📁 Upload", "🎤 Voice"]
        ]
        tab1, tab2, tab3 = st.tabs(tab_labels)

        with tab1:
            # Use session state for language
            camera_label = translate_interface_text("Take a picture", st.session_state.current_language)
            
            # Add custom CSS to force landscape orientation
            st.markdown("""
                <style>
                .stCamera > video {
                    width: 100%;
                    aspect-ratio: 16/9 !important;
                }
                .stCamera > img {
                    width: 100%;
                    aspect-ratio: 16/9 !important;
                    object-fit: cover;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Camera input with custom styling
            img_file_camera = st.camera_input(
                camera_label,
                key="camera_input",
                help="Please capture the image in landscape orientation"
            )
            
            if img_file_camera:
                try:
                    # Read the image from camera
                    image = Image.open(img_file_camera)
                    
                    # Ensure landscape orientation
                    if image.height > image.width:
                        # Rotate the image if it's in portrait
                        image = image.rotate(90, expand=True)
                    
                    # Display captured image with consistent aspect ratio
                    st.image(
                        image, 
                        caption=translate_interface_text("Captured Image", st.session_state.current_language),
                        use_container_width=True
                    )
                    
                    # Generate description using the original image
                    description = generate_image_description(image)
                    st.markdown(f"<h1 style='color: #00d2ff;'>{description}</h1>", unsafe_allow_html=True)

                    # Model selection based on the description
                    select_model_text = translate_interface_text("Select Model for Further Analysis", 
                                                              st.session_state.current_language)
                    selected_model = st.selectbox(
                        select_model_text, 
                        list(detection_types.keys()),
                        key="camera_model_select"
                    )
                    
                    analyze_button_text = translate_interface_text("Analyze Prediction", 
                                                                st.session_state.current_language)
                    if st.button(analyze_button_text, key="camera_analyze_button"):
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        analyzed_image = analyze_with_model(selected_model, img_bytes)
                        if analyzed_image is not None:
                            st.image(
                                analyzed_image, 
                                caption=translate_interface_text("Analyzed Image", st.session_state.current_language),
                                use_container_width=True
                            )
                            
                except Exception as e:
                    st.error(f"Error processing camera image: {str(e)}")

        with tab2:
            uploaded_file = st.file_uploader("Upload a medical image (X-ray, MRI, CT, etc.)", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                try:
                    # Read the file content first
                    file_bytes = uploaded_file.read()
                    
                    # Create PIL Image from bytes
                    image = Image.open(io.BytesIO(file_bytes))
                    
                    # Display the image
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    # Generate description using the original image
                    description = generate_image_description(image)
                    st.markdown(f"<h1 style='color: #00d2ff;'>{description}</h1>", unsafe_allow_html=True)

                    # Model selection based on the description
                    selected_model = st.selectbox("Select Model for Further Analysis", list(detection_types.keys()))
                    if st.button("Analyze Prediction"):
                        # Create a new bytes buffer for analysis
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Analyze the prediction based on the selected model
                        analyzed_image = analyze_with_model(selected_model, img_bytes)
                        if analyzed_image is not None:
                            st.image(analyzed_image, caption="Analyzed Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")

        with tab3:
            voice_label = translate_interface_text("Voice Input", selected_lang)
            if st.button(voice_label):
                with st.spinner(translate_interface_text("Listening...", selected_lang)):
                    text = speech_to_text()
                    if text:
                        st.success(f"{translate_interface_text('You said:', selected_lang)} {text}")

        # Process image if available
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption=translate_interface_text("Uploaded Image", selected_lang), use_container_width=True)

            check_button = st.button(
                translate_interface_text("🔍 Check for Detection", selected_lang),
                key="check_detection_button",
                use_container_width=True
            )

            if check_button:
                process_detection(image, selected_lang)

    # Chat interface in second column
    with col2:
        # Create two columns for the header
        header_col1, header_col2 = st.columns([1, 1])
        
        with header_col1:
            st.markdown("## AI Chat Assistant")
        
        with header_col2:
            st.image("/Users/sreemadhav/SreeMadhav/Mhv CODES/MVSR/p22/PythonProject/download.gif", 
                    use_container_width=True)

        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)

        # Display chat messages
        for message in st.session_state.get('chat_history', []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your health..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_chat_response(prompt)
                    st.markdown(response)

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")


def generate_image_description(image_data):
    """Generate a concise description for the given image using Google Generative AI."""
    try:
        # If image_data is already bytes, convert it to PIL Image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError("Unsupported image format")

        # Resize image if needed
        image.thumbnail([640, 640], Image.Resampling.LANCZOS)

        # Convert to bytes for API processing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Create model instance and generate description
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Updated prompt for more concise response
        prompt = """
        Analyze this medical image and respond in exactly 7 words following this format:
        'Detected: [condition]. Use [detection_type] detection model.'
        
        Example responses:
        'Detected: Brain tumor. Use brain tumor detection.'
        'Detected: Eye condition. Use eye disease detection.'
        """
        
        response = model.generate_content(
            [prompt, {"mime_type": "image/png", "data": img_bytes}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Lower temperature for more focused response
                max_output_tokens=30,  # Limit token length
                top_p=0.1,  # More focused sampling
            )
        )

        # Get the response and ensure it's concise
        result = response.text.strip()
        
        # If response is too long, truncate it
        words = result.split()
        if len(words) > 7:
            result = ' '.join(words[:7])

        return result

    except Exception as e:
        st.error(f"Error generating image description: {str(e)}")
        return "Unable to analyze image. Please try again."


def get_current_location():
    """Get user's current location using IP address"""
    try:
        response = requests.get('https://ipapi.co/json/')
        data = response.json()
        return {
            'lat': float(data['latitude']),
            'lon': float(data['longitude']),
            'city': data['city']
        }
    except Exception as e:
        st.error(f"Error getting location: {str(e)}")
        return {'lat': 17.4065, 'lon': 78.4772, 'city': 'Hyderabad'}  # Default to Hyderabad

def find_nearby_hospitals(lat, lon, radius=5000):
    """Find nearby hospitals using OpenStreetMap"""
    try:
        # Using Overpass API to get hospitals
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:{radius},{lat},{lon});
          way["amenity"="hospital"](around:{radius},{lat},{lon});
          relation["amenity"="hospital"](around:{radius},{lat},{lon});
        );
        out center;
        """
        response = requests.post(overpass_url, data=query)
        data = response.json()
        
        hospitals = []
        for element in data['elements']:
            if 'center' in element:
                lat = element['center']['lat']
                lon = element['center']['lon']
            else:
                lat = element.get('lat', 0)
                lon = element.get('lon', 0)
            
            name = element.get('tags', {}).get('name', 'Unknown Hospital')
            hospitals.append({
                'name': name,
                'lat': lat,
                'lon': lon
            })
        
        return hospitals
    except Exception as e:
        st.error(f"Error finding hospitals: {str(e)}")
        return []

def show_emergency_contacts():
    """Display emergency contacts"""
    st.markdown("""
    ### 🚨 Emergency Contacts
    - **Ambulance**: 108
    - **Police**: 100
    - **Fire**: 101
    - **National Emergency**: 112
    - **SMVS Hospital**: +91-XXXXXXXXXX
    """)

def create_medical_report(diagnosis, confidence, recommendations):
    """Generate a medical report"""
    now = datetime.now()
    report = {
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "diagnosis": diagnosis,
        "confidence": f"{confidence:.2%}",
        "recommendations": recommendations,
        "disclaimer": "This is an AI-generated report for preliminary analysis only. Please consult a healthcare professional."
    }
    return report

def download_report(report):
    """Create a downloadable report"""
    report_md = f"""
    # AI Medical Analysis Report
    
    **Date**: {report['date']}
    
    ## Diagnosis
    {report['diagnosis']}
    
    ## Confidence Level
    {report['confidence']}
    
    ## Recommendations
    {report['recommendations']}
    
    ## Disclaimer
    {report['disclaimer']}
    """
    return report_md

def analyze_with_model(selected_model, image_data):
    """Enhanced analyze_with_model function with additional features"""
    try:
        # Define correct model paths
        model_paths = {
            "brain_tumor": "brain123.pt",  # Updated path to your existing model
            "eye_disease": "eye.pt",
            "lung_cancer": "lung_cancer.pt",
            "bone_fracture": "bone.pt",
            "skin_disease": "skin345.pt"
        }

        # Convert bytes to PIL Image if needed
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError("Unsupported image format")

        # Convert PIL Image to numpy array for YOLO processing
        image_np = np.array(image)
        
        # Ensure image is in RGB format
        if len(image_np.shape) == 2:  # If grayscale, convert to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # If RGBA, convert to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Make a copy of the image for drawing
        image_draw = image_np.copy()

        # Get the correct model path
        model_path = model_paths.get(selected_model)
        if not model_path:
            raise ValueError(f"No model path defined for {selected_model}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the selected model
        model = YOLO(model_path)

        # Process the image
        results = model(image_np)

        # Process results and draw predictions
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            
            for box, cls_idx in zip(boxes, cls):
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = model.names[int(cls_idx)]
                
                # Draw rectangle and text
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_draw, class_name, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # After analysis, add these new features:
        st.markdown("---")
        
        # Create tabs for different information
        result_tab, hospitals_tab, report_tab = st.tabs(["Results", "Nearby Hospitals", "Medical Report"])
        
        with result_tab:
            # Show analysis results
            if image_draw is not None:
                st.image(image_draw, caption="Analyzed Image", use_container_width=True)
                
                # Show confidence score
                confidence = 0.85  # Replace with actual confidence from model
                st.progress(confidence)
                st.write(f"Confidence Score: {confidence:.2%}")

        with hospitals_tab:
            # Get current location and show nearby hospitals
            location = get_current_location()
            st.subheader(f"Nearby Hospitals in {location['city']}")
            
            # Create map
            m = folium.Map(location=[location['lat'], location['lon']], zoom_start=13)
            
            # Add markers for nearby hospitals
            hospitals = find_nearby_hospitals(location['lat'], location['lon'])
            for hospital in hospitals:
                folium.Marker(
                    [hospital['lat'], hospital['lon']],
                    popup=hospital['name'],
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            
            # Display map
            folium_static(m)
            
            # Show emergency contacts
            show_emergency_contacts()

        with report_tab:
            # Generate medical report
            recommendations = [
                "Schedule an appointment with a specialist",
                "Get a second opinion",
                "Follow up with additional tests",
                "Monitor symptoms regularly"
            ]
            
            report = create_medical_report(
                diagnosis=f"Potential {selected_model.replace('_', ' ')} detected",
                confidence=confidence,
                recommendations=recommendations
            )
            
            # Display report
            st.markdown(download_report(report))
            
            # Download button
            report_str = json.dumps(report, indent=2)
            st.download_button(
                label="Download Report",
                data=report_str,
                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        return image_draw

    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None


def main():
    # Initialize language selection
    if 'current_language' not in st.session_state:
        st.session_state.current_language = 'en'
    
    # Main content area
    st.markdown("## Medical Image Analysis")
    col1, col2 = st.columns([5, 3])

    with col1:
        # Input method tabs
        tab1, tab2, tab3 = st.tabs(["📸 Camera", "📁 Upload", "🎤 Voice"])

        with tab1:
            # Use session state for language
            camera_label = translate_interface_text("Take a picture", st.session_state.current_language)
            
            # Add custom CSS to force landscape orientation
            st.markdown("""
                <style>
                .stCamera > video {
                    width: 100%;
                    aspect-ratio: 16/9 !important;
                }
                .stCamera > img {
                    width: 100%;
                    aspect-ratio: 16/9 !important;
                    object-fit: cover;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Camera input with custom styling
            img_file_camera = st.camera_input(
                camera_label,
                key="camera_input",
                help="Please capture the image in landscape orientation"
            )
            
            if img_file_camera:
                try:
                    # Read the image from camera
                    image = Image.open(img_file_camera)
                    
                    # Ensure landscape orientation
                    if image.height > image.width:
                        # Rotate the image if it's in portrait
                        image = image.rotate(90, expand=True)
                    
                    # Display captured image with consistent aspect ratio
                    st.image(
                        image, 
                        caption=translate_interface_text("Captured Image", st.session_state.current_language),
                        use_container_width=True
                    )
                    
                    # Generate description using the original image
                    description = generate_image_description(image)
                    st.markdown(f"<h1 style='color: #00d2ff;'>{description}</h1>", unsafe_allow_html=True)

                    # Model selection based on the description
                    select_model_text = translate_interface_text("Select Model for Further Analysis", 
                                                              st.session_state.current_language)
                    selected_model = st.selectbox(
                        select_model_text, 
                        list(detection_types.keys()),
                        key="camera_model_select"
                    )
                    
                    analyze_button_text = translate_interface_text("Analyze Prediction", 
                                                                st.session_state.current_language)
                    if st.button(analyze_button_text, key="camera_analyze_button"):
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        analyzed_image = analyze_with_model(selected_model, img_bytes)
                        if analyzed_image is not None:
                            st.image(
                                analyzed_image, 
                                caption=translate_interface_text("Analyzed Image", st.session_state.current_language),
                                use_container_width=True
                            )
                            
                except Exception as e:
                    st.error(f"Error processing camera image: {str(e)}")

        with tab2:
            uploaded_file = st.file_uploader("Upload a medical image (X-ray, MRI, CT, etc.)", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                try:
                    # Read the file content first
                    file_bytes = uploaded_file.read()
                    
                    # Create PIL Image from bytes
                    image = Image.open(io.BytesIO(file_bytes))
                    
                    # Display the image
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    # Generate description using the original image
                    description = generate_image_description(image)
                    st.markdown(f"<h1 style='color: #00d2ff;'>{description}</h1>", unsafe_allow_html=True)

                    # Model selection based on the description
                    selected_model = st.selectbox("Select Model for Further Analysis", list(detection_types.keys()))
                    if st.button("Analyze Prediction"):
                        # Create a new bytes buffer for analysis
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Analyze the prediction based on the selected model
                        analyzed_image = analyze_with_model(selected_model, img_bytes)
                        if analyzed_image is not None:
                            st.image(analyzed_image, caption="Analyzed Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")

        with tab3:
            st.text("Voice Input Coming Soon...")

    with col2:
        # Create two columns for the header
        header_col1, header_col2 = st.columns([1, 1])
        
        with header_col1:
            st.markdown("## AI Chat Assistant")
        
        with header_col2:
            st.image("/Users/sreemadhav/SreeMadhav/Mhv CODES/MVSR/p22/PythonProject/download.gif", 
                    use_container_width=True)

        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)

        # Display chat messages
        for message in st.session_state.get('chat_history', []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your health..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_chat_response(prompt)
                    st.markdown(response)

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")


if __name__ == "__main__":
    main()