from flask import Flask, request, render_template
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)
    
    def forward(self, x):
        x = self.base_model(x)
        return torch.sigmoid(x)

model = ImageClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the model
model.load_state_dict(torch.load('improved_model.pth'))


def allowed_file(filename):
    """
    Check if a file has an allowed extension.

    Parameters:
    -----------
    filename : str
        The name of the file to check.

    Returns:
    --------
    bool
        True if the file has an allowed extension, False otherwise.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(file_path):
    model.eval()
    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.round(output).item()
    return "Real" if prediction == 1.0 else "Fake", output.item()

# instantiate app w/ config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model_path = 'deepfake_detector_model.keras'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='no file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='no selected file')
        if file and allowed_file(file.filename):
            # save the uploaded file to the uploads directory
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            # predict if the image is Real or Fake
            prediction, prediction_percentage = predict_image(filename)
            # clean up the uploaded file
            os.remove(filename)
            # determine result message
            # result = 'Fake' if prediction >= 0.5 else 'Real'
            # render result to the user
            return render_template('index.html', result=prediction, prediction_percentage=f"{prediction_percentage:.4f}")
        else:
            return render_template('index.html', error='allowed file types are png, jpg, jpeg')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)