import gradio as gr
import numpy as np
import random
import subprocess
from PIL import Image

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "https://huggingface.co/RuiTenkawa/Hasku/blob/main/hassakuXLSfwNsfw_betaV06.safetensors"  # Replace to the model you would like to use

if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024
# pipe = False # For local debug
# device = "cpu"
# v 2.5 not perfect but idgf
version = "v2.5"
Image_Storage = []

def Download_Model(link):
    subprocess.run(["curl", "-Lo", "ManualDownload.safetensors", link])
    return "Completed"

def Load_Model(value):
    global pipe
    if value == "HassakuSFW":
        pipe = StableDiffusionXLPipeline.from_single_file(model_repo_id, torch_dtype=torch_dtype, token="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")
    else:
        pipe = StableDiffusionXLPipeline.from_single_file("/content/ManualDownload.safetensors", torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention()
    global imgpipe
    imgpipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    return {prompt: gr.Text(placeholder="Enter your prompt", interactive=True), PipeReady: gr.Checkbox(value=True)}

def Device():
    if device == "cuda":
        return True
    else:
        return False

def Handle_Upload(value):
    Image_Storage.append(Image.fromarray(value))
    return Image_Storage

def Handle_Images(index):
    intind = int(index)
    if (intind < 0):
        return Image_Storage
    else:
        del Image_Storage[intind-1]
        return Image_Storage

def Swap_pipes(evt: gr.SelectData):
    global state
    if evt.value == "Text2Img":
        state = "Text2Img" 
        return {
            width: gr.Slider(visible=True),
            height: gr.Slider(visible=True),
            strength: gr.Slider(visible=False),
        }
    elif evt.value == "Img2Img":
        state = "Img2Img"
        return {
            width: gr.Slider(visible=False),
            height: gr.Slider(visible=False),
            strength: gr.Slider(visible=True),
        }
    return {
            width: gr.Slider(visible=False),
            height: gr.Slider(visible=False),
            strength: gr.Slider(visible=False),
        }

def infer(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    Image2Img,
    strength,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    if (state == "Text2Img"):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
    elif (state == "Img2Img"):
        image = imgpipe(
            prompt=prompt,
            image=Image_Storage[int(Image2Img) - 1],
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        ).images[0]

    Image_Storage.append(image)

    return image, Handle_Images(-1), seed

css = """
#col-container {
    margin: 0 auto;
    max-width: 756px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):

        with gr.Tab("Text2Img") as Tx2i:
            gr.Text(visible=False, label="Curiosity kills")
                
        with gr.Tab("Img2Img") as I2i:
            Image_2img = gr.Text(
                label="Img2Img",
                show_label=False,
                max_lines=1,
                placeholder="Номер кортынки",
                container=False,
            )
        
        with gr.Tab("More") as Moreomore:
                with gr.Row():
                    PipeReady = gr.Checkbox(value=False, interactive=False, label="Model loaded")
                    gr.Checkbox(value=Device(), interactive=False, label="Cuda enabled")
                    gr.Checkbox(interactive=False, label="Version: " + version)
                with gr.Row():
                    Drop = gr.Dropdown(["HassakuSFW", "Manual Download"], label="Model", interactive=True)
                    load_button = gr.Button("Load", scale=0, variant="primary")
                with gr.Row():
                    downloadlink = gr.Text(
                    label="Download link",
                    show_label=False,
                    max_lines=1,
                    placeholder="https://civitai.com/api/download/models/378499?token=YOURTOKEN",
                    container=False,
                    interactive=True,
                )
                    down_button = gr.Button("Download", scale=0, variant="primary")
        
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Firstly load model in More",
                    container=False,
                    interactive=False,
                )
            run_button = gr.Button("Gen", scale=0, variant="primary")
        
        result = gr.Image(label="Result", show_label=False, interactive=True)

        with gr.Row():
            Image_Del_Num = gr.Text(
                label="Удаление кортынки",
                show_label=False,
                max_lines=1,
                placeholder="Номер кортынки. Мне лень танцевать с бубном поэтому удаляем картинки из галереи по номеру",
                container=False,
            )
            Del_button = gr.Button("Delete", scale=0, variant="primary")

        gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[5], rows=[1], height=150)
        
        with gr.Accordion("Advanced Settings", open=True):
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                value="worst quality, low quality, normal quality, bad anatomy, bad hands,",
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width", # Limited as for Nvidia T4, more will cause memory overflow
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,  # Replace with defaults that work for your model
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,  # Replace with defaults that work for your model
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=15.0,
                    step=0.1,
                    value=7.0,  # Replace with defaults that work for your model
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=70,
                    step=1,
                    value=20,  # Replace with defaults that work for your model
                )
            with gr.Row():
                strength = gr.Slider(
                    label="Strength",
                    minimum=0,
                    maximum=1,
                    step=0.02,
                    value=0.8,  # Replace with defaults that work for your model
                )
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            Image_2img,
            strength,
        ],
        outputs=[result, gallery, seed],
    )
    gr.on(
        triggers=[Del_button.click, Image_Del_Num.submit],
        fn=Handle_Images,
        inputs=[
            Image_Del_Num,
        ],
        outputs=[gallery],
    )
    gr.on(
        triggers=[Tx2i.select, Moreomore.select, I2i.select],
        fn=Swap_pipes,
        inputs=[],
        outputs=[width, height, strength],
    )
    gr.on(
        triggers=[result.upload],
        fn=Handle_Upload,
        inputs=[
           result,
        ],
        outputs=[gallery],
    )
    gr.on(
        triggers=[load_button.click],
        fn=Load_Model,
        inputs=[
           Drop,
        ],
        outputs=[prompt, PipeReady],
    )
    gr.on(
        triggers=[down_button.click],
        fn=Download_Model,
        inputs=[
           downloadlink,
        ],
        outputs=[downloadlink],
    )

if __name__ == "__main__":
    demo.launch()