import numpy as np
from PIL import Image
import tensorflow as tf
import google.generativeai as genai

# ✅ Load the trained DR model once
model = tf.keras.models.load_model("retina_model10.h5")

# ✅ Gemini API configuration (use your new key here)
genai.configure(api_key="your key")  # Replace with your new API key

# ✅ Preprocess the retina image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Predict DR class from retina image
def predict_image(image: Image.Image):
    processed = preprocess_image(image)
    predictions = model.predict(processed)
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    return class_names[np.argmax(predictions)]

# ✅ Get response from Gemini chatbot
def get_chatbot_response(user_query: str):
    system_prompt = """
    You are an AI assistant specialized in answering queries related to Diabetic Retinopathy Detection, CNN models, and medical image analysis.
    If the user asks anything unrelated to these topics, politely decline to answer.and answer the query in short 3-4 lines
    """

    try:
        gemini = genai.GenerativeModel("gemini-2.5-flash")
        response = gemini.generate_content([system_prompt, user_query])
        return response.text or "I'm here to assist with project-related queries only."
    except Exception as e:
        return f"Error from Gemini API: {str(e)}"
