import streamlit as st
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import io
from datetime import datetime

from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText


from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Load pre-trained model
model = tf.keras.models.load_model('wildfire_detection_project/wildfire_detection.h5')

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize image to model's input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict whether the image is a wildfire
def predict_wildfire(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction[0][0]  # Returning the first prediction (assuming binary classification)

def send_sms_alert():
    account_sid = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    auth_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body="🔥 Wildfire detected! Immediate action required.",
        from_='XXXXXXXXXXXX',
        to='XXXXXXXXXXXXXX'
    )
    print(f"SMS sent: {message.sid}")

# Function to send email alert
def send_email_alert():
    sender_email = "XXXXXXXXXXXXXXXXXXXXXXXXXXX"
    receiver_email = "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    password = "XXXXXXXXXXXXXXXXXXXXXXxXXX"

    subject = "🔥 Wildfire Detected Alert"
    body = "A wildfire has been detected in the uploaded image. Immediate action is recommended."

    # Compose the email
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="JPEG")
    img_buffer.seek(0)

    # Attach the image
    image = MIMEImage(img_buffer.read(), name='wildfire_image.jpg')
    msg.attach(image)


    # Send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

    print("Email alert sent successfully!")



# Streamlit UI
st.set_page_config(page_title="Wildfire Detection", page_icon="🔥", layout="wide")

# Custom CSS for background, text, and buttons
st.markdown(
    """
    <style>
    body {
        background-image: url('https://i.pinimg.com/736x/52/e0/df/52e0df8dc145703e2946dce2682f2f8c.jpg');
        background-size: cover;
        color: white;
    }
    .stButton>button {
        background-color: #FF4500;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #FF6347;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #FF4500;
    }
    .tagline {
        font-size: 22px;
        font-style: italic;
        color: #FFD700;
    }
    .info-box {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True
)

# Title and tagline
st.markdown("<h1 class='title'>Wildfire Detection <span style='font-size: 28px; color: #FFD700;'>🔥</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>Predicting Wildfires in Images with Artificial Intelligence 🌍</p>", unsafe_allow_html=True)

# Description about the system
st.markdown("""
    <div class="info-box">
        <h3>System Description:</h3>
        <p>This system uses a trained Convolutional Neural Network (CNN) model to detect the presence of wildfires in images. 
        Simply upload an image, and the system will predict if there is a wildfire or not. The system is trained with high-accuracy 
        using real-world wildfire images to ensure reliable results.</p>
    </div>
    """, unsafe_allow_html=True)

# Dark & Light Mode Toggle
#if st.button("Toggle Dark/Light Mode"):
#    st.session_state.dark_mode = not st.session_state.get("dark_mode", False)

# Upload image
uploaded_file = st.file_uploader("Choose an image to check for wildfire 🌟", type=["jpg", "png", "jpeg","jfif"])

if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_container_width=True , width=100)
    
    # Predict button
    if st.button('Predict 🔥'):
        result = predict_wildfire(img)
        current_time = datetime.now().strftime('%Y-%m-%d%d %H:%M:%S')
        
        # Show prediction result
        if result > 0.5:
            st.success("Wildfire detected! 🔥", icon="🔥")
            st.write(f"**Date & Time of Prediction:** {current_time}")
            
            send_sms_alert()
            send_email_alert()
        else:
            st.success("No wildfire detected. 🌱", icon="🌱")


