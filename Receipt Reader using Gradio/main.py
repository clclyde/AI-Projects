import gradio as gr
import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en', 'tl'], gpu=True)  # Use GPU if available, otherwise set gpu=False

# Optional: Preprocess image for better OCR accuracy
def preprocess_image(img):
    img = ImageOps.grayscale(img)  # Convert to grayscale
    img = ImageOps.autocontrast(img)  # Enhance contrast
    img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.Resampling.LANCZOS)  # Resize for better OCR
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  # Sharpen the image
    return img

"""
# Group text by row based on y-coordinates
def group_text_by_row(results):
    rows = []
    for bbox, text, confidence in results:
        # Extract the y-coordinate of the bounding box's center
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        # Check if the text belongs to an existing row
        added_to_row = False
        for row in rows:
            # If the y-center is close to an existing row's y-center, add the text to that row
            if abs(row['y_center'] - y_center) < 10:  # Adjust the threshold as needed
                row['texts'].append(text)
                added_to_row = True
                break
        # If not added to any row, create a new row
        if not added_to_row:
            rows.append({'y_center': y_center, 'texts': [text]})
    # Sort rows by their y-coordinates and join texts in each row
    rows.sort(key=lambda r: r['y_center'])
    return [" ".join(row['texts']) for row in rows]
"""

def group_text_by_row(results):
    rows = []
    for bbox, text, confidence in results:
        # Extract the y-coordinate of the bounding box's center
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_start = bbox[0][0]  # Extract the x-coordinate of the bounding box's start
        # Check if the text belongs to an existing row
        added_to_row = False
        for row in rows:
            # If the y-center is close to an existing row's y-center, add the text to that row
            if abs(row['y_center'] - y_center) < 10:  # Adjust the threshold as needed
                row['texts'].append((x_start, text))
                added_to_row = True
                break
        # If not added to any row, create a new row
        if not added_to_row:
            rows.append({'y_center': y_center, 'texts': [(x_start, text)]})
    # Sort rows by their y-coordinates
    rows.sort(key=lambda r: r['y_center'])
    # Sort texts within each row by their x-coordinates and join them with spaces
    formatted_rows = []
    for row in rows:
        row['texts'].sort(key=lambda t: t[0])  # Sort by x_start
        # Add spaces between texts based on the distance between x-coordinates
        row_text = ""
        prev_x = 0
        for x_start, text in row['texts']:
            if row_text:  # Add spaces based on the gap between x-coordinates
                gap = int((x_start - prev_x) / 10)  # Adjust the divisor for spacing
                row_text += " " * max(gap, 1)
            row_text += text
            prev_x = x_start + len(text) * 10  # Approximate the end x-coordinate
        formatted_rows.append(row_text)
    return formatted_rows

# Main function to analyze the receipt
def analyze_receipt(image):
    # Preprocess the image
    img = preprocess_image(image)
    
    # Perform OCR with bounding box details
    results = reader.readtext(np.array(img), detail=1, paragraph=False)  # Enable detail=1 for bounding box info
    
    # Group text by row
    grouped_text = group_text_by_row(results)
    
    # Join rows with newlines
    extracted_text = "\n".join(grouped_text)
    print("Extracted Text:\n", extracted_text)  # Debug: Print the extracted text with new lines
    
    return extracted_text

# Gradio UI
demo = gr.Interface(
    fn=analyze_receipt,
    inputs=gr.Image(type="pil", label="Upload Receipt Image"),
    outputs=gr.Textbox(label="Extracted Text", lines=10),  # Ensure multiline support with `lines`
    title="Receipt Text Extractor (EasyOCR)",
    description="Upload a receipt image to extract all text. Text on the same row in the receipt will appear on the same row in the output."
)

demo.launch(share=True)