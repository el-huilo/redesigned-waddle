import gradio as gr
import numpy as np
import random, os, subprocess, torch, sys, requests
from PIL import Image

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerAncestralDiscreteScheduler
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, GGUFQuantizationConfig
from diffusers.utils import export_to_gif

models_link_list = [
    "cleanedUp"
]
models_name_list = [
    "cleanedUp"
]
models_types_list = [
    "SDXL",
    "SD",
    "HunyuanVideo"
]
class AuxVars:
    def __init__(self):
        self.version = "v3.7"
        self.was_loaded = False
        self.animiter = 0
        self.AnimpipeReady = False
        self.T2V = False
        self.max_image_size = 1024
        self.max_frames = 196
        self.min_frames = 2
        self.step_frames = 1
        if torch.cuda.is_available():
            self.torch_dtype = torch.float16
            self.device = "cuda"
        else:
            self.torch_dtype = torch.float32
            self.device = "cpu"
    def sl(self):
        self.theme = None
    def sdev(self):
        self.version = "dev"
        self.theme = gr.themes.Soft(primary_hue=gr.themes.colors.violet, secondary_hue=gr.themes.colors.neutral, neutral_hue=gr.themes.colors.sky)

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
            self.animpipe.enable_free_noise(context_length=16, context_stride=4)
            # self.animpipe.enable_free_noise_split_inference() why torch.tensor error
            # self.animpipe.unet.enable_forward_chunking(16)
            aux.AnimpipeReady = True
            self.animpipe.to(aux.device)
        aux.was_loaded = True
    
    def HunLoad(self, pipe):
        self.pipe = pipe
        self.pipe.enable_vae_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()
        aux.was_loaded = True
        aux.AnimpipeReady = True
        aux.T2V = True
        aux.max_frames = 129
        aux.min_frames = 5
        aux.step_frames = 4

MAX_SEED = np.iinfo(np.int32).max
Image_Storage = []
Prompt_Storage = {}
aux = AuxVars()
pipes = Pipes()
try:
    mode = sys.argv[1]
except:
    aux.sl()
else:
    if mode == 'dev_build':
        aux.sdev()
    else:
        aux.sl()
if not os.path.isdir("/content/gifs"):
    os.mkdir("/content/gifs")

def Download_Model(link):
    subprocess.run(["curl", "-Lo", "Manual_Download.safetensors", link])
    return "Finished"

def DownNload_Model(value, type):
    if value == "Manual Download":
        print("\n\nYou are kidding right?\n\n")
        return
    else:
        ModelLink = models_link_list[models_name_list.index(value)]
        ModelPath = value.replace(" ", "_")
        subprocess.run(["curl", "-Lo", f"{ModelPath}.safetensors", ModelLink])
        return Load_Model(value, type)

def Load_Model(value, type):
    ModelPath = value.replace(" ", "_")
    if type == "SDXL/Pony":
        pipes.load(StableDiffusionXLPipeline.from_single_file(f"/content/{ModelPath}.safetensors", torch_dtype=aux.torch_dtype), type)
    elif type == "SD":
        pipes.load(StableDiffusionPipeline.from_single_file(f"/content/{ModelPath}.safetensors", torch_dtype=aux.torch_dtype), type)
    else:
        transformer = HunyuanVideoTransformer3DModel.from_single_file("https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q4_K_M.gguf",
                                                                      quantization_config=GGUFQuantizationConfig(),
                                                                      )
        pipes.HunLoad(HunyuanVideoPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                                                    transformer=transformer,
                                                    torch_dtype=aux.torch_dtype))
    return update_all()

def check_version():
    response = requests.get("https://github.com/el-huilo/redesigned-waddle/releases/latest")
    if response.url.split('/').pop() == aux.version:
        return gr.Checkbox(value=True, label="Version: " + aux.version + " Latest")
    else:
        return gr.Checkbox(label="Version: " + aux.version + " New available")

def update_all():
    if aux.was_loaded == True and aux.T2V == False:
        return {prompt: gr.Text(placeholder="Enter your prompt", interactive=aux.was_loaded),
                PipeReady: gr.Checkbox(value=aux.was_loaded),
                gallery: Image_Storage,
                VStatus: check_version(),
                gif_button: gr.Button(visible=aux.AnimpipeReady),
                num_frames: gr.Slider(visible=aux.AnimpipeReady, maximum=aux.max_frames, minimum=aux.min_frames, step=aux.step_frames),
                fps_count: gr.Slider(visible=aux.AnimpipeReady),
                Tx2v: gr.Tab(visible=False),
                FrameKey: gr.Text(visible=aux.AnimpipeReady),
                PromptsDict: gr.Accordion(visible=aux.AnimpipeReady)
                }
    elif aux.was_loaded == True and aux.T2V == True:
        return {prompt: gr.Text(placeholder="Firstly load model in More", interactive=aux.was_loaded),
                PipeReady: gr.Checkbox(value=aux.was_loaded),
                gallery: Image_Storage,
                VStatus: check_version(),
                gif_button: gr.Button(visible=aux.AnimpipeReady),
                num_frames: gr.Slider(visible=aux.AnimpipeReady),
                fps_count: gr.Slider(visible=aux.AnimpipeReady),
                Tx2v: gr.Tab(visible=True),
                FrameKey: gr.Text(visible=aux.AnimpipeReady),
                PromptsDict: gr.Accordion(visible=aux.AnimpipeReady)
                }
    else:
        return {prompt: gr.Text(placeholder="Firstly load model in More", interactive=aux.was_loaded),
                PipeReady: gr.Checkbox(value=aux.was_loaded),
                gallery: Image_Storage,
                VStatus: check_version(),
                gif_button: gr.Button(visible=aux.AnimpipeReady),
                num_frames: gr.Slider(visible=aux.AnimpipeReady),
                fps_count: gr.Slider(visible=aux.AnimpipeReady),
                Tx2v: gr.Tab(visible=False),
                FrameKey: gr.Text(visible=aux.AnimpipeReady),
                PromptsDict: gr.Accordion(visible=aux.AnimpipeReady)
                }

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

def Handle_MultPrompts(key, value, called_by):
    if called_by == "Add":
        Prompt_Storage[int(key)] = value
    else:
        del Prompt_Storage[int(key)]
    return "\n".join(f"{a}: {Prompt_Storage[a]}" for a in Prompt_Storage)

def Swap_pipes(evt: gr.SelectData):
    global state
    if evt.value == "Text2Img":
        state = "Text2Img" 
        return {
            width: gr.Slider(visible=True),
            height: gr.Slider(visible=True),
            strength: gr.Slider(visible=False),
            negative_prompt: gr.Text(visible=True),
            AdvSet: gr.Accordion(visible=True),
        }
    elif evt.value == "Img2Img":
        state = "Img2Img"
        return {
            width: gr.Slider(visible=False),
            height: gr.Slider(visible=False),
            strength: gr.Slider(visible=True),
            negative_prompt: gr.Text(visible=True),
            AdvSet: gr.Accordion(visible=True),
        }
    elif evt.value == "More":
        return {
            width: gr.Slider(visible=False),
            height: gr.Slider(visible=False),
            strength: gr.Slider(visible=False),
            negative_prompt: gr.Text(visible=False),
            AdvSet: gr.Accordion(visible=False),
        }
    else:
        state = "Text2Vid"
        return {
            width: gr.Slider(visible=True),
            height: gr.Slider(visible=True),
            strength: gr.Slider(visible=False),
            negative_prompt: gr.Text(visible=False),
            AdvSet: gr.Accordion(visible=True),
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
    if width == 1024 and height == 1024:
        width = 768
        height = 768
        print("1024x1024 too much. Changing to 768x768")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    if state == "Text2Vid":
        image = pipes.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            num_frames=num_frames,
        ).frames[0]
    else:
        if Prompt_Storage:
            prompt = Prompt_Storage
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

with gr.Blocks(css=css, theme=aux.theme) as demo:
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
        with gr.Tab("Text2Vid", visible=False) as Tx2v:
            gr.Text(visible=False, label="Curiosity kills")
        with gr.Tab("More") as Moreomore:
                with gr.Row():
                    PipeReady = gr.Checkbox(value=False, interactive=False, label="Model loaded")
                    gr.Checkbox(value=Device(), interactive=False, label="Cuda enabled")
                    VStatus = gr.Checkbox(interactive=False)
                with gr.Row():
                    Drop = gr.Dropdown(models_name_list, label="Model", interactive=True, min_width=200, scale=2)
                    TypeDrop = gr.Dropdown(models_types_list, label="Type", interactive=True, scale=2)
                    with gr.Column(min_width=50, scale=1):
                        load_button = gr.Button("Load", scale=0, variant="primary")
                        loadDown_button = gr.Button("Down/Load", scale=0, variant="primary")
                with gr.Row():
                    downloadlink = gr.Text(
                    show_label=False,
                    max_lines=1,
                    placeholder="https://civitai.com/api/download/models/000000?token=YOURTOKEN",
                    container=False,
                    interactive=True,
                )
                    down_button = gr.Button("Download", scale=0, variant="primary")
        
        with gr.Row():
            FrameKey = gr.Text(
                    show_label=False,
                    max_lines=1,
                    placeholder="Frame:",
                    container=False,
                    interactive=True,
                    scale=0,
                    min_width=100,
                    visible=False
                )
            prompt = gr.Text(
                    show_label=False,
                    max_lines=3,
                    placeholder="Firstly load model in More",
                    container=False,
                    interactive=False,
                    scale=8,
                )
            run_button = gr.Button("Gen", scale=0, variant="primary", min_width=100)
            gif_button = gr.Button("Gif", scale=0, variant="primary", visible=aux.AnimpipeReady, min_width=100)
        with gr.Accordion("Prompts dictionary for gif", open=False, visible=False) as PromptsDict:
            with gr.Row():
                add_prompt_button = gr.Button("Add", scale=1, variant="primary")
                del_prompt_button = gr.Button("Del", scale=1, variant="primary")
            promptlist = gr.Text(
                show_label=False,
                placeholder="Frame : prompt",
                container=False,
                interactive=False,
            )
        result = gr.Image(label="Result", show_label=False, interactive=True)

        with gr.Row():
            Image_Del_Num = gr.Text(
                show_label=False,
                max_lines=1,
                placeholder="Номер кортынки. all удалить все",
                container=False,
            )
            Del_button = gr.Button("Delete", scale=0, variant="primary")

        gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[5], rows=[1])
        
        with gr.Accordion("Advanced Settings", open=True, visible=False) as AdvSet:
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
                    visible=False,
                )
                num_frames = gr.Slider(
                    label="Frames count",
                    minimum=2,
                    maximum=32,
                    step=1,
                    value=9,
                    visible=False,
                )
                fps_count = gr.Slider(
                    label="gif fps",
                    minimum=2,
                    maximum=32,
                    step=1,
                    value=8,
                    visible=False,
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
        triggers=[add_prompt_button.click],
        fn=Handle_MultPrompts,
        inputs=[
            FrameKey,
            prompt,
            add_prompt_button,
        ],
        outputs=[promptlist],
    )
    gr.on(
        triggers=[del_prompt_button.click],
        fn=Handle_MultPrompts,
        inputs=[
            FrameKey,
            prompt,
            del_prompt_button,
        ],
        outputs=[promptlist],
    )
    gr.on(
        triggers=[Tx2i.select, Moreomore.select, I2i.select, Tx2v.select],
        fn=Swap_pipes,
        inputs=[],
        outputs=[width, height, strength, negative_prompt, AdvSet],
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
        outputs=[prompt, PipeReady, gallery, VStatus, gif_button, num_frames, fps_count, Tx2v, FrameKey, PromptsDict],
    )
    gr.on(
        triggers=[loadDown_button.click],
        fn=DownNload_Model,
        inputs=[
           Drop,
           TypeDrop
        ],
        outputs=[prompt, PipeReady, gallery, VStatus, gif_button, num_frames, fps_count, Tx2v, FrameKey, PromptsDict],
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
        outputs=[prompt, PipeReady, gallery, VStatus, gif_button, num_frames, fps_count, Tx2v, FrameKey, PromptsDict],
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
