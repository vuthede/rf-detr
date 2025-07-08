import os
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests
import io

url = "/media/vuthede/Lexar/data/oms_incabin/images/Leopard_250523_9/images_RGB/captures_LI-OX05B1S_005690.raw.png"
image = Image.open(url)

model = get_model("rfdetr-base")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_image_base.jpg")

