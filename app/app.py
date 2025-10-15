import os

from starlette.staticfiles import StaticFiles

parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

import gradio
import torch
from fastapi import FastAPI, Request
from starlette.templating import Jinja2Templates


from lama_inpaint import build_lama_model
from segment_anything import sam_model_registry, SamPredictor

from remove_anything_UI import demo as remove
from fill_anything_UI import demo as fill
from replace_anything_UI import demo as replace
from remove_anything_video_UI import demo as remove_video

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(parent_directory, "templates"))


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# build models
model = {}
# build the sam model
model_type = "vit_b"
file_path = os.path.join(parent_directory, "pretrained_models/sam_vit_b_01ec64.pth")
ckpt_p = file_path
model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam.to(device=device)
model['sam'] = SamPredictor(model_sam)

# build the lama model
lama_config = os.path.join(parent_directory, "lama/configs/prediction/default.yaml")
lama_ckpt = os.path.join(parent_directory, "pretrained_models/big-lama")
device = "cuda" if torch.cuda.is_available() else "cpu"
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

app.mount("/static", StaticFiles(directory=os.path.join(parent_directory, "templates/static")), name="static")
app = gradio.mount_gradio_app(app, remove(model), path="/removeAnything")
app = gradio.mount_gradio_app(app, fill(model), path="/fillAnything")
app = gradio.mount_gradio_app(app, replace(model), path="/replaceAnything")
app = gradio.mount_gradio_app(app, remove_video(model), path="/removeAnythingVideo")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
