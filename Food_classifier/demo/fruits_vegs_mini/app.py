import torch
import gradio as gr
from model import load_model, predict  # Import the functions
from PIL import Image

# Load the model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model = load_model("Food_classifier/demo/fruits_vegs_mini/effnetb1_6_epochs.pth", device=device)

def predict_gradio(image):
    """
    Wrapper around the predict function to format results
    for Gradio's output (label + top probabilities, etc.).
    """
    pred_dict, pred_time = predict(image, loaded_model, device=device)
    # Return the top 3 predictions for the Label output
    return pred_dict, pred_time

# Create a few example images or load from a local folder
example_images = [
    ["Food_classifier/demo/fruits_vegs_mini/examples/Image_1.jpg"],
    ["Food_classifier/demo/fruits_vegs_mini/examples/Image_6.jpg"],
    ["Food_classifier/demo/fruits_vegs_mini/examples/Image_10.jpg"]
]

demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3),
        gr.Number(label="Prediction Time (s)")
    ],
    examples=example_images,
    title="Fruit and Vegetable Recognition"
)

demo.launch()
