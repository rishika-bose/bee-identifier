from fastai.vision.all import *
import gradio as gr

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('model.pkl')

pathlib.PosixPath = temp

im_bee = Image.open('bee.jpg')
im_wasp = Image.open('wasp.jpg')
im_mixed = Image.open('mixed.jpg')

im_bee.thumbnail((192,192))
im_wasp.thumbnail((192,192))
im_mixed.thumbnail((192,192))

categories = ('bee','wasp')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

examples = ['bee.jpg','wasp.jpg','mixed.jpg']

intf = gr.Interface(
    fn = classify_image,
    inputs = gr.Image(),
    outputs = gr.Textbox(),
    examples = examples
)

intf.launch(inline=False)