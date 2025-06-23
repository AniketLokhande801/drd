# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import google.generativeai as genai  # Import Gemini API

# # Configure Gemini API Key
# genai.configure(api_key="AIzaSyAFVwoLGtLmnxr9naSmqF36fXiihYSYqKc")

# # Load the trained model
# model = tf.keras.models.load_model('retina_model10.h5')

# # Function to preprocess the image
# def preprocess_image(image):
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

# # Function for chatbot response using Gemini API
# def get_chatbot_response(user_query):
#     system_prompt = """
#     You are an AI assistant specialized in answering queries related to Diabetic Retinopathy Detection, CNN models, and medical image analysis.
#     If the user asks anything unrelated to these topics, politely decline to answer.
#     """
#     model = genai.GenerativeModel("gemini-pro")
#     response = model.generate_content([system_prompt, user_query])
    
#     return response.text if response.text else "I'm here to assist with project-related queries only."

# # Streamlit app
# st.title('Diabetic Retinopathy Detection using CNN')
# st.write('Upload an image of the retina to get the prediction.')

# # Image Upload and Prediction
# uploaded_image = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
# if uploaded_image is not None:
#     image = Image.open(uploaded_image)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
    
#     image_array = preprocess_image(image)
#     predictions = model.predict(image_array)
#     class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
#     predicted_class = class_names[np.argmax(predictions)]
    
#     st.subheader(f'Prediction: {predicted_class}')

# # Chatbot Section
# st.subheader("Ask the AI Chatbot")
# st.write("Ask project-related questions about Diabetic Retinopathy Detection and CNN models.")

# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_input = st.text_input("Type your question here...", key="chat_input")

# if st.button("Ask"):
#     if user_input:
#         chatbot_response = get_chatbot_response(user_input)
#         st.session_state.chat_history.append(("You", user_input))
#         st.session_state.chat_history.append(("AI", chatbot_response))

# # Display chat history
# for role, message in st.session_state.chat_history:
#     st.write(f"**{role}:** {message}")




from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from util import predict_image, get_chatbot_response
from PIL import Image
import io

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict_image(image)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result
    })

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request, "response": ""})

@app.post("/chat", response_class=HTMLResponse)
async def chatbot(request: Request, question: str = Form(...)):
    response = get_chatbot_response(question)
    return templates.TemplateResponse("chatbot.html", {"request": request, "response": response, "question": question})
