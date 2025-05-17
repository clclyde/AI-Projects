import cv2
import numpy as np
from PIL import Image
import paddleocr
import gradio as gr
from google import genai
import tempfile
import os

def preprocess_image(image):
       # Convert the image to grayscale.
       image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
       # Apply Gaussian blur to reduce noise.
       image = cv2.GaussianBlur(image, (5, 5), 0)
       # Apply thresholding to highlight text.
       _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
       # Convert back to a Pillow image.
       image = Image.fromarray(image)
       return image

ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', det_model_dir='path/to/detection/model', rec_model_dir='path/to/recognition/model')

GOOGLE_API_KEY = "AIzaSyB2XILc8usKmH2tj76oeTANfGdt4vFgxRc"
# genai.configure(api_key=GOOGLE_API_KEY)  # Removed configure
client = genai.Client(api_key=GOOGLE_API_KEY) # Changed to Client

def paddle_receipt(receipt):
    result = ""
    if receipt is None:
        return "There is no image uploaded"
    try:
        read = ocr.ocr(receipt, cls=True)  # Perform OCR
        if read and len(read) > 0 and len(read[0]) > 0:
            counter = 1
            for i in read[0]:
                if counter == 5:
                    result += "\n"
                    counter = 1
                result += i[1][0] + " "
                counter += 1
            return result  # Return extracted text if successful
        else:
            return "No text detected"  # Return message if no text detected
    except Exception as e:
        return f"Error during OCR: {e}"  # Return error message if exception occurs


def analyze_receipt_with_gemini(user_prompt, ocr_text=None):
    prompt = f"""
    Analyze the following text from a receipt and extract key information,
    including the store name, store address, date, total amount, change received,
    money given to the cashier, and any items purchased.
    If possible, also identify the payment method.  Provide the output in a conversational format.

    Receipt Text:
    ```{ocr_text}```
    """
    if ocr_text:
        # If ocr_text is provided, include it in the prompt
        prompt = f"""
        {user_prompt}

        Receipt Text:
        ```{ocr_text}```
        """
    else:
        # If no ocr_text is provided, use the user_prompt
        prompt = user_prompt
    try:
        response = client.models.generate_content(  # Use client.generate_content
            model="gemini-2.0-flash",  # Explicitly specify the model.
            contents=prompt, # changed the contents
        )
        return response.text if response.text else "No information extracted."
    except Exception as e:
        return f"Error during Gemini analysis: {e}"

def process_image_and_chat(image, chat_history):
    if image is None:
        return "Please upload a receipt image.", chat_history

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        image.save(temp_image.name)
        image_path = temp_image.name

    ocr_text = paddle_receipt(temp_image.name)
    os.unlink(temp_image.name)
    if "Error:" in ocr_text:
        chat_history.append((None, ocr_text))
        return chat_history  # Return the chat_history, which now contains the error

    # Pass the ocr_text for initial analysis
    gemini_response = analyze_receipt_with_gemini("Analyze this receipt:", ocr_text)
    chat_history.append(("", gemini_response))  # Append as (user_msg, bot_response)
    return chat_history  # Return the updated chat history


def launch_chatbot():
    with gr.Blocks() as app:
        gr.Markdown("## Receipt Analysis Chatbot")

        # Chatbot and input are defined *within* the Blocks context
        chatbot = gr.Chatbot(height=300)
        user_input = gr.Textbox(label="Enter your message") # Added user input textbox
        image_input = gr.Image(label="Upload Receipt Image", type="pil")
        chat_state = gr.State([])

        # Function to handle user text input
        def respond_to_text(user_message, chat_history):
            if user_message:
                # Pass user message directly to Gemini for analysis
                gemini_response = analyze_receipt_with_gemini(user_message, chat_history)
                chat_history.append((user_message, gemini_response))
            return chat_history, chat_history

        # Set the .on method for the user_input.
        user_input.submit(
            fn=respond_to_text,
            inputs=[user_input, chat_state],
            outputs=[chatbot, chat_state]
        )

        # Define a function to handle button clicks.
        def on_image_upload(image, chat_history):
            updated_history = process_image_and_chat(image, chat_history)
            return updated_history, updated_history

        # Set the .on method for the image_input.
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input, chat_state],
            outputs=[chatbot, chat_state]
        )

    app.launch()


if __name__ == "__main__":
    launch_chatbot()
