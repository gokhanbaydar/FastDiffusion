import gradio as gr
from core import sdxl_styles, generator

fd = generator.FastDiffusionGenerator()


def generate(
    prompt,
    negative_prompt,
    style,
    controlnet_conditioning_scale,
    control_schema,
    control_image,
):
    if not fd.models_loaded:
        fd.load_models()
    fd.controlnet_conditioning_scale = controlnet_conditioning_scale
    style = sdxl_styles.styles.get(style)
    prompt = style[0].replace("{prompt}", prompt)
    negative_prompt = negative_prompt + "," + style[1]
    return fd.run(control_image, prompt, negative_prompt, control_schema)


prompt = gr.Textbox(lines=3)
negative_prompt = gr.Textbox(lines=3)
style = gr.Dropdown(choices=sdxl_styles.style_keys)
control_schema = gr.Radio(["None", "canny", "depth"])
controlnet_conditioning_scale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5)
control_image = gr.Image(type="pil")


demo = gr.Interface(
    fn=generate,
    inputs=[
        prompt,
        negative_prompt,
        style,
        controlnet_conditioning_scale,
        control_schema,
        control_image,
    ],
    outputs="image",
)
demo.launch()
