import os, torch, json
import clip
from PIL import Image
from torchvision.datasets import CIFAR100
from flask import Flask, render_template, request

app = Flask(__name__)
device = "cpu"  # if torch.cuda.is_available() else "cpu"
# Load model cls
model, preprocess = clip.load('ViT-B/32', device)
print("{} mode loaded done".format('ViT-B/32'))
path = ''


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/blog')
def blog():
    return render_template("blog.html")


@app.route('/check', methods=['POST'])
def check():
    file = request.files['inputFile']
    global path
    path = "static/temp/" + file.filename
    image_data = file.save(path)
    # Setup
    torch.cuda.empty_cache()
    print("{} is setup done".format(device))

    # Load dictionary
    text = list()
    with open('model\dictionary.txt', 'r') as f:
        text = json.loads(f.read())

    print("dictionary of {}  loaded done".format(len(text)))

    # text to weight
    text_inputs = torch.cat([clip.tokenize(c) for c in text]).to(device)
    text_features = model.encode_text(text_inputs)
    print("dictionary to torch done")
    image_input = preprocess(Image.open(path)).unsqueeze(0).to(device)
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    sen = text[indices[0]]
    return render_template('index.html', image=path, sen=sen, score=int(values[0].tolist() * 100))


@app.route('/predict', methods=["GET", "POST"])
def predit():
    global path
    torch.cuda.empty_cache()
    print(request.form['sen'])
    text_inputs = torch.cat([clip.tokenize(request.form.get('sen'))]).to(device)
    text_features = model.encode_text(text_inputs)
    print("dictionary to torch done")
    print(path)
    image_input = preprocess(Image.open(path)).unsqueeze(0).to(device)
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    return render_template('index.html', image=path, sen=request.form['sen'], score=int(values[0].tolist() * 100))


app.run(debug=True)
