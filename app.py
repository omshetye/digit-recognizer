from flask import Flask, request, render_template, jsonify
import base64
from model import FFNN, CNN
from PIL import Image
from io import BytesIO
import torch
import torchvision
import cv2


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

cnn = CNN()
device = torch.device('cpu')  # Load model on CPU
cnn.load_state_dict(torch.load('mnist_model_cnn.pth', map_location=device))
cnn.eval()  # Set model to evaluation mode


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict_digit():
    try:
        data = request.get_json(silent=True)
        model_choice = data['model']
        image = request.get_json(silent=True)['image'].split(",")[1]
        image_data = base64.urlsafe_b64decode(image)
        # Convert to PIL Image
        img = Image.open(BytesIO(image_data))
        img = img.convert('L')
        # Preprocess the image (resize, normalize, convert to tensor)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        img_tensor = transform(img)  # Add batch dimension
        print(img_tensor.shape)
        if model_choice == 'ffnn':
            confidence, prediction = ffnn.predict(img_tensor)
        elif model_choice == 'cnn':
            confidence, prediction = cnn.predict(img_tensor)
            print(prediction)
        confidence = float(confidence.item())
        print(prediction)
        response = {
            "prediction": str(prediction),
            "confidence": str(confidence)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)
