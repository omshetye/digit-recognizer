from flask import Flask, request, render_template, jsonify
import base64
from model import FFNN
from PIL import Image
from io import BytesIO
import torch
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np

input_size = 784
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
num_classes = 10

app = Flask(__name__)

ffnn = FFNN(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
device = torch.device('cpu')  # Load model on CPU
ffnn.load_state_dict(torch.load('mnist_model.pth', map_location=device))
ffnn.eval()  # Set model to evaluation mode

onnx_model_path = 'cnn_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"]) 

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess_image(image_data):
    image_data = base64.urlsafe_b64decode(image_data.split(",")[1])
    img = Image.open(BytesIO(image_data))
    img = img.convert('L')  # Convert to grayscale
    
    # Preprocess the image (resize, normalize, convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_with_onnx(img_tensor):
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}
    ort_outs = ort_session.run([ort_session.get_outputs()[0].name], ort_inputs)
    return ort_outs

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict_digit():
    try:
        data = request.get_json(silent=True)
        model_choice = data['model']
        image_data = data['image']
        
        # Preprocess the image and convert to tensor
        img_tensor = preprocess_image(image_data)

        if model_choice == 'ffnn':
            # Predict using FFNN model
            confidence, prediction = ffnn.predict(img_tensor)
        
        elif model_choice == 'cnn':
            # Predict using CNN model (ONNX)
            ort_outs = predict_with_onnx(img_tensor)
            prediction = np.argmax(ort_outs[0])
            confidence = np.max(ort_outs[0])

        # Prepare response
        confidence = float(confidence.item())
        response = {
            "prediction": str(prediction),
            "confidence": str(confidence)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
