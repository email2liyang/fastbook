import gradio as gr
from fastai.vision.all import *

learn = load_learner("export.pkl")

labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


title = "Bear Breed Classifier"
description = "A bear breed classifier trained on from internet bears 'grizzly','black','teddy' with fastai. Created as a demo for Gradio and HuggingFace Spaces."
interpretation = "default"
enable_queue = True
gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
).launch(share=False)
