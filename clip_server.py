import clip
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model, preprocess = clip.load("ViT-B/32", device="cpu")

@app.post("/embed")
def embed():
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vector = model.encode_image(image_tensor).tolist()[0]

    return jsonify(vector)
if __name__ == "__main__":
    # Port 10000 trở lên để Render tự map, lấy từ env
    import os
    port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=5001)
