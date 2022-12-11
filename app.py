from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import pipeline
import json
import torch
import os
from flask import Flask, request
from googletrans import Translator

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")
    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.y(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

app = Flask(__name__)


@app.route('/predecirFoto', methods = ['POST'])
def reponderFoto():
    if request.files['file'].filename == '':
      return 'No selected file'
    print(request.files['file'].filename)
    imagen = Image.open(request.files['file'].stream)
    nombreImagen = request.files['file'].filename
    print(nombreImagen)
    guardar = imagen.save(nombreImagen)
    response =  predict_step([nombreImagen])
    os.remove(nombreImagen)
    translator = Translator()
    translator.raise_Exception = True
    translation = translator.translate(response[0], src="en", dest="es")
    print(translation.text)
    return json.dumps(translation.text, ensure_ascii=False)

@app.route('/predecirEmociones', methods = ['GET'])
def responderTexto():
    comment = request.args.get('comment')
    translator = Translator()
    translator.raise_Exception = True
    translation = translator.translate(comment, src="es", dest="en")
    respuesta = classifier(translation.text)
    print(translation.text)
    return json.dumps(respuesta[0])


app.run(debug=True)
