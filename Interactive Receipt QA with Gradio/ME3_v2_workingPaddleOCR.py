from paddleocr import PaddleOCR
import gradio as gr
import easyocr
# Initialize PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def paddle_receipt(receipt):
  result = ""
  read = ocr.ocr(receipt, cls=True)
  if receipt is None:
    return "There is no image uploaded"
  counter = 1
  for i in read[0]:
    if counter==5:
      result+="\n"
      counter=1
    result+=i[1][0] + " "
    counter+=1
  if result is None:
    return "No text detected"
  return result

#app = gr.Interface(easyocr_receipt, inputs="image", outputs="text", title = "Extracting Text from Receipts using EasyOCR")
#app.launch()
with gr.Blocks() as app:
    gr.Markdown("## Image to Text")
    with gr.Tab("PaddleOCR"):
        image_input = gr.Image()
        output_paddle = gr.Textbox()
        b1 = gr.Button("Extract Receipt Info (PaddleOCR)")
        b1.click(paddle_receipt, inputs=image_input, outputs=output_paddle)
app.launch()

