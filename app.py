import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename

import torchxrayvision as xrv
import skimage
import torch
import torchvision
from llama_cpp import Llama
import requests
from tqdm import tqdm


model_path = './models/'
model_name = 'Qwen2-1.5b-it-bioinstruct.Q8_0.gguf'
# Check if the model directory exists, if not, create it
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Check if the model file exists, if not, download it
model_file_path = os.path.join(model_path, model_name)
if not os.path.exists(model_file_path):
    url = "https://huggingface.co/RichardErkhov/ehristoforu_-_Qwen2-1.5b-it-bioinstruct-gguf/resolve/main/Qwen2-1.5b-it-bioinstruct.Q8_0.gguf?download=true"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(model_file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")
model_path = model_path + model_name
assistant_text = "Reply in short key points only."
chat_history = []

app = Flask(__name__)
app.secret_key = "some_secret_key"


# upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def build_prompt(history):
    """
    Build a prompt from the chat history.
    Format: each turn is prefixed with the speaker, and the prompt ends expecting the assistant's answer.
    """
    prompt = ""
    for turn in history:
        prompt += f"{turn['role']}: {turn['message']}\n"
    prompt += assistant_text + ": "
    return prompt

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/chat', methods=['POST'])
def chat():
    llm = Llama(model_path=model_path)
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    # Build the prompt
    prompt = f"{user_input}\n{assistant_text}"

    # Generate a response using the Llama model
    response = llm(prompt, max_tokens=150, repeat_penalty=10)

    return jsonify({'response': response['choices'][0]['text']})


@app.route("/chatbox")
def chatbox():
    return render_template("chatbox.html")


### X-RAY/MRI IMAGE CLASSIFICATION FUNCTIONALITY ###
def classify_xray(filename: str) -> dict:
    # Read the image
    img = skimage.io.imread(filename)
    # Normalize from 0..255 to -1024..1024
    img = xrv.datasets.normalize(img, 255)
    # Convert to single color channel
    img = img.mean(2)[None, ...]  # Ensure shape (1, H, W) for grayscale

    # Apply transformation: center crop & resize to 224x224
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)

    # Convert to PyTorch tensor with batch dimension
    img = torch.from_numpy(img).unsqueeze(0)

    # Load pre-trained DenseNet model
    model = xrv.models.DenseNet(weights="densenet121-res224-all")

    # Run inference
    outputs = model(img)

    # Convert results to a dictionary {pathology: float_value}
    results = {
        pathology: float(value)
        for pathology, value in zip(model.pathologies, outputs[0].detach().numpy())
    }
    return results


@app.route("/upload", methods=["POST"])
def upload_file():
    if "xray_image" not in request.files:
        return "No file part in the request", 400

    file = request.files["xray_image"]
    if file.filename == "":
        return "No file selected", 400

    # Secure the filename and save to UPLOAD_FOLDER
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Classify the X-ray
    results = classify_xray(filepath)

    # Store results in session (alternative: use a database)
    session["classification_results"] = results
    session["uploaded_image"] = filepath  # Store the file path for displaying

    return redirect(url_for("results"))


@app.route("/results")
def results():
    results = session.get("classification_results", {})
    image_path = session.get("uploaded_image", None)

    return render_template("results.html", results=results, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
