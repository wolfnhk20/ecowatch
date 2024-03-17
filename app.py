import io
from flask import Flask, request, render_template, jsonify
from PIL import Image

from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD-FOLDER'] = 'uploads'
model = YOLO('best.pt')

classNames = ['Iguana', 'Indian Elephant', 'Indian Wolf', 'Tiger', 'White Tiger', 'Antelope', 'Asiatic Lion', 'Barking Deer', 'Bengal Tiger', 'Bengal Tiger', 'Asiatic Black Bear', 'Blackbuck', 'Chimpanzee', 'Gharial', 'Indian Bison', 'Indian Rock Python', 'Jackal', 'King Cobra', 'leopard', 'lion-tailed macaque', 'monkey', 'nilgiri tahr', 'one horned rhino', 'orangutan', 'peacock', 'porcupine', 'red panda', 'indian rhinoceros', 'sambar deer', 'sloth bear', 'snow leopard', 'indian star tortoise']

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/identify", methods=["POST"])
def identify_species():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file.filename.endswith('.jpg' or '.png' or '.gif' or '.jpeg'):
        image_data = file.read()
        img = Image.open(io.BytesIO(image_data))
        results = model(img, save=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                print("Detected -->", classNames[cls])

    response = {
        "label": classNames[cls],
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
