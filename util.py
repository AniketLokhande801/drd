# # import numpy as np
# # from PIL import Image
# # import tensorflow as tf
# # import google.generativeai as genai

# # # Load the DR model once
# # model = tf.keras.models.load_model("retina_model10.h5")

# # # Gemini API config
# # genai.configure(api_key="AIzaSyAFVwoLGtLmnxr9naSmqF36fXiihYSYqKc")

# # def preprocess_image(image: Image.Image):
# #     image = image.resize((224, 224))
# #     image = np.array(image) / 255.0
# #     image = np.expand_dims(image, axis=0)
# #     return image

# # def predict_image(image: Image.Image):
# #     processed = preprocess_image(image)
# #     predictions = model.predict(processed)
# #     class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
# #     return class_names[np.argmax(predictions)]

# # def get_chatbot_response(user_query):
# #     system_prompt = """
# #     You are an AI assistant specialized in answering queries related to Diabetic Retinopathy Detection, CNN models, and medical image analysis.
# #     If the user asks anything unrelated to these topics, politely decline to answer.
# #     """
# #     gemini = genai.GenerativeModel("gemini-pro")
# #     response = gemini.generate_content([system_prompt, user_query])
# #     return response.text if response.text else "I'm here to assist with project-related queries only."


# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_groq import ChatGroq  # Groq LangChain integration

# # Set API key (replace with your actual key or use env var)
# os.environ["GROQ_API_KEY"] = "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"

# # Load the DR model once
# model = tf.keras.models.load_model("retina_model10.h5")

# def preprocess_image(image: Image.Image):
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

# def predict_image(image: Image.Image):
#     processed = preprocess_image(image)
#     predictions = model.predict(processed)
#     class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
#     return class_names[np.argmax(predictions)]

# # LangChain + Groq Chat Model (You can choose 'mixtral-8x7b', 'llama3-70b-8192', etc.)
# chat_model = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="mixtral-8x7b-32768",  # or llama3-70b-8192
#     temperature=0.4
# )

# def get_chatbot_response(user_query):
#     messages = [
#         SystemMessage(content="""
#         You are an AI assistant specialized in Diabetic Retinopathy Detection,medical image analysis.
#         Kindly decline unrelated queries.
#         """),
#         HumanMessage(content=user_query)
#     ]
    
#     response = chat_model.invoke(messages)
#     return response.content if response.content else "I'm here to assist with project-related queries only."

import numpy as np
from PIL import Image
import tensorflow as tf
import google.generativeai as genai

# ✅ Load the trained DR model once
model = tf.keras.models.load_model("retina_model10.h5")

# ✅ Gemini API configuration (use your new key here)
genai.configure(api_key="AIzaSyCcltcssEhI5OiaEHFt0ansD8WrSqO99zo")  # Replace with your new API key

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
