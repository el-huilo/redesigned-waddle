import gradio as gr
import numpy as np
import random, os, subprocess, torch
from PIL import Image

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import export_to_gif

model_repo_id = "https://huggingface.co/RuiTenkawa/Hasku/blob/main/hassakuXLSfwNsfw_betaV06.safetensors"  # Replace to the model you would like to use

class AuxVars:
    def __init__(self):
        self.version = "v3.1"
        self.was_loaded = False
        self.animiter = 0
        self.AnimpipeReady = False
        self.max_image_size = 1024
        if torch.cuda.is_available():
            self.torch_dtype = torch.float16
            self.device = "cuda"
        else:
            self.torch_dtype = torch.float32
            self.device = "cpu"

class Pipes:
    def load(self, pipe, type):
        self.pipe = pipe
        self.pipe = self.pipe.to(aux.device)
        if torch.cuda.is_available():
            self.pipe.enable_xformers_memory_efficient_attention()
        if type == "SDXL/Pony":
            self.imgpipe = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
            aux.AnimpipeReady = False
        else:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            self.imgpipe = StableDiffusionImg2ImgPipeline(**self.pipe.components)
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
            self.animpipe = AnimateDiffPipeline.from_pipe(self.pipe, motion_adapter=adapter)
            self.animpipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.animpipe.scheduler.config, beta_schedule="linear")
            aux.AnimpipeReady = True
            self.animpipe.to(aux.device)
        aux.was_loaded = True

MAX_SEED = np.iinfo(np.int32).max
Image_Storage = []
aux = AuxVars()
pipes = Pipes()
if not os.path.isdir("/content/gifs"):
    os.mkdir("/content/gifs")

def Download_Model(link):
    subprocess.run(["curl", "-Lo", "ManualDownload.safetensors", link])
    return "Finished"

def Load_Model(value, type):
    if type == "SDXL/Pony":
        if value == "HassakuSFW":
            pipes.load(StableDiffusionXLPipeline.from_single_file(model_repo_id, torch_dtype=aux.torch_dtype, token="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"), type)
        else:
            pipes.load(StableDiffusionXLPipeline.from_single_file("/content/ManualDownload.safetensors", torch_dtype=aux.torch_dtype), type)
    else:
        pipes.load(StableDiffusionPipeline.from_single_file("/content/ManualDownload.safetensors", torch_dtype=aux.torch_dtype), type)
    return {prompt: gr.Text(placeholder="Enter your prompt", interactive=True), PipeReady: gr.Checkbox(value=True)}

def check_version():
    # works only in google colab
    if os.environ.get('LTS_V') == aux.version:
        return gr.Checkbox(value=True, label="Version: " + aux.version + " Latest")
    else:
        return gr.Checkbox(label="Version: " + aux.version + " New available")

def update_all():
    if aux.was_loaded == True:
        return {prompt: gr.Text(placeholder="Enter your prompt", interactive=aux.was_loaded), PipeReady: gr.Checkbox(value=aux.was_loaded), gallery: Image_Storage, VStatus: check_version(), gif_button: gr.Button(visible=aux.AnimpipeReady)}
    else:
        return {prompt: gr.Text(placeholder="Firstly load model in More", interactive=aux.was_loaded), PipeReady: gr.Checkbox(value=aux.was_loaded), gallery: Image_Storage, VStatus: check_version(), gif_button: gr.Button(visible=aux.AnimpipeReady)}

def Device():
    if aux.device == "cuda":
        return True
    else:
        return False

def Handle_Upload(value):
    Image_Storage.append(Image.fromarray(value))
    return Image_Storage

def Handle_Images(index):
    if index == 'all':
        del Image_Storage[0:]
        return Image_Storage
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
            width: gr.Slider(visible=True, maximum=aux.max_image_size),
            height: gr.Slider(visible=True, maximum=aux.max_image_size),
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
        image = pipes.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
    elif (state == "Img2Img"):
        image = pipes.imgpipe(
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

def animinfer(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    num_frames,
    fps_count,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    image = pipes.animpipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        num_frames=num_frames,
    ).frames[0]
    export_to_gif(image, f"gifs/{aux.animiter}.gif", fps=fps_count)
    Image_Storage.append(f"gifs/{aux.animiter}.gif")
    aux.animiter += 1
    return Image_Storage[-1], Handle_Images(-1), seed

css = """
#col-container {
    margin: 0 auto;
    max-width: 756px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):

        with gr.Tab("Text2Img") as Tx2i:
            with gr.Row():
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
                    VStatus = gr.Checkbox(interactive=False)
                with gr.Row():
                    Drop = gr.Dropdown(["HassakuSFW", "Manual Download"], label="Model", interactive=True)
                    TypeDrop = gr.Dropdown(["SDXL/Pony", "SD"], label="Type", interactive=True)
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
            gif_button = gr.Button("Gif", scale=0, variant="primary", visible=aux.AnimpipeReady)
        
        result = gr.Image(label="Result", show_label=False, interactive=True)

        with gr.Row():
            Image_Del_Num = gr.Text(
                label="Удаление кортынки",
                show_label=False,
                max_lines=1,
                placeholder="Номер кортынки. all удалить все",
                container=False,
            )
            Del_button = gr.Button("Delete", scale=0, variant="primary")

        gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[5], rows=[1])
        
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
                    maximum=aux.max_image_size,
                    step=32,
                    value=512,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=aux.max_image_size,
                    step=32,
                    value=512,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=50.0,
                    step=0.1,
                    value=7.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=70,
                    step=1,
                    value=20,
                )
            with gr.Row():
                strength = gr.Slider(
                    label="Strength",
                    minimum=0,
                    maximum=1,
                    step=0.02,
                    value=0.8,
                )
                num_frames = gr.Slider(
                    label="Frames count",
                    minimum=2,
                    maximum=32,
                    step=1,
                    value=8,
                )
                fps_count = gr.Slider(
                    label="gif fps",
                    minimum=3,
                    maximum=32,
                    step=1,
                    value=8,
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
           TypeDrop
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
    gr.on(
        triggers=[Moreomore.select],
        fn=update_all,
        inputs=[],
        outputs=[prompt, PipeReady, gallery, VStatus, gif_button],
    )
    gr.on(
        triggers=[gif_button.click],
        fn=animinfer,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            num_frames,
            fps_count,
        ],
        outputs=[result, gallery, seed],
    )
if __name__ == "__main__":
    demo.launch()