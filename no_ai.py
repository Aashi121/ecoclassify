import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="♻️ Waste Classifier",
    page_icon="🗑️",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('keras_model.h5')

model = load_model()

# --- Waste Categories ---
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# --- Sustainability Tips & Ideas ---
reuse_ideas = {
    'cardboard': [
        "Recycled into packaging, cartons, or even insulation.",
        "Used in DIY crafts or composting.",
        "Makes great mulch for gardening!"
    ],
    'glass': [
        "Melted into new bottles or tiles.",
        "Used in decorative art and construction blocks.",
        "Crushed into sand substitute for roads."
    ],
    'metal': [
        "Reused for car parts, tools, or electronics.",
        "Melted into cans or structural beams.",
        "Used in bike frames or home decor."
    ],
    'paper': [
        "Made into tissue paper or egg cartons.",
        "Used for notebooks, newspapers, or recycled art.",
        "Composted into garden fertilizer."
    ],
    'plastic': [
        "Converted into furniture or textile fibers.",
        "Used to build eco-friendly roads.",
        "Made into toys or household items."
    ],
    'trash': [
        "Used in waste-to-energy conversion.",
        "Processed for landfill gas recovery.",
        "Replaced by reusable alternatives to reduce future waste."
    ]
}

tips = [
    "Always clean and dry your recyclables before disposal.",
    "Separate organic and inorganic waste at the source.",
    "Try to reduce single-use plastics in your daily life.",
    "Use reusable bags and bottles whenever possible.",
    "Start composting organic waste at home!"
]

# --- Title & Description ---
st.markdown("<h1 style='text-align: center;'>♻️ Smart Waste Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of waste to identify its type and get recycling ideas.</p>", unsafe_allow_html=True)

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload your waste image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Display Image ---
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # --- Preprocess Image ---
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Predict ---
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    waste_type = class_names[class_index].lower()

    # --- Display Prediction ---
    st.subheader(f"🔍 Prediction: **{class_names[class_index]}**")
    st.metric(label="🔒 Confidence", value=f"{confidence:.2%}")
    st.progress(confidence)

    # --- What Can Be Done With This Waste? ---
    st.markdown("### 💡 What Can Be Done With This Waste?")
    st.info(random.choice(reuse_ideas[waste_type]))

    # --- Sustainability Tip ---
    st.markdown("---")
    st.markdown("### 🌿 Sustainability Tip")
    st.success(random.choice(tips))

    # --- Call to Action ---
    st.markdown("---")
    st.markdown("🤖 **Next Step?**")
    st.markdown("- Integrate this app with IoT-based smart bins!")
    st.markdown("- Add cloud logging to track and report types of waste in your area.")
