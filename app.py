
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm
import io
from rembg import remove

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=6, embed_dim=768)

model_path = 'gemstones_model_without_early(11).pth'  # Update the path to where your model is saved
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


class_names = ['Emerald', 'Fake Emerald', 'Fake Ruby', 'Fake Turquoise', 'Ruby', 'Turquoise']
confidence_threshold = 0.6  
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    # Remove the background
    image = remove(image)
    image = image.convert('RGB')
    image.save("backRemoved.png")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        softmax_probs = F.softmax(outputs, dim=1)
        max_prob, predicted_class = torch.max(softmax_probs, 1)
        max_prob = max_prob.item()
        predicted_class = predicted_class.item()

 
    if max_prob < confidence_threshold:
        return jsonify({'type': 'Uncertain', 'confidence': max_prob})

   
    response_class = class_names[predicted_class]

  

    return jsonify({'type': response_class, 'confidence': max_prob})

if __name__ == '__main__':
    app.run(debug=True)
