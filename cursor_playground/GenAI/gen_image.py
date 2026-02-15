import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from random import randint

# 1. Load the model with the best settings for 12GB VRAM
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

# 2. Optimization: Use the DPM-Solver++ (Fastest & Sharpest)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 3. VRAM Optimization for Cursor/Dev environment
pipe.enable_attention_slicing()

STYLE_PRESETS = {
    "None": "",
    "Cinematic": "cinematic lighting, dramatic composition, ultra detailed, film still",
    "Anime": "anime style, clean line art, vibrant colors, studio quality",
    "Photorealistic": "photorealistic, highly detailed, natural lighting, 8k",
    "Fantasy Art": "fantasy concept art, epic, painterly, intricate details",
    "Cyberpunk": "cyberpunk style, neon lights, futuristic city, moody atmosphere",
}


def build_styled_prompt(prompt, preset):
    prompt = (prompt or "").strip()
    style_text = STYLE_PRESETS.get(preset, "")
    if not style_text:
        return prompt
    if not prompt:
        return style_text
    return f"{prompt}, {style_text}"


def generate(prompt, preset, randomize_seed, seed, steps, guidance, width, height):
    prompt_text = build_styled_prompt(prompt, preset)
    if not prompt_text:
        raise gr.Error("Please enter a prompt before generating.")

    used_seed = randint(0, 2**31 - 1) if randomize_seed else int(seed)
    generator = torch.Generator(device="cuda").manual_seed(used_seed)

    image = pipe(
        prompt=prompt_text,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=int(width),
        height=int(height),
        generator=generator,
    ).images[0]
    status = (
        f"Seed: `{used_seed}` | Preset: `{preset}` | "
        f"Steps: `{int(steps)}` | Guidance: `{guidance}` | "
        f"Size: `{int(width)}x{int(height)}`"
    )
    return image, status, prompt_text

# 4. Build the Gradio UI
custom_css = """
.app-shell {
    max-width: 1180px;
    margin: 0 auto;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    opacity: 0.85;
    margin-bottom: 1rem;
}
.control-card, .preview-card {
    border: 1px solid rgba(125, 125, 125, 0.24);
    border-radius: 14px;
    padding: 14px;
    background: rgba(255, 255, 255, 0.02);
}
"""

with gr.Blocks(
    title="SD 1.5 Local Studio",
    theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"),
    css=custom_css,
) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <div class="hero-title">SD 1.5 Local Studio</div>
            <div class="hero-subtitle">
                Professional prompt-to-image dashboard powered by your local Stable Diffusion model.
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=4, elem_classes=["control-card"]):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A futuristic robot coding in a neon-lit workspace, highly detailed",
                    lines=4,
                )
                preset_dropdown = gr.Dropdown(
                    label="Style Preset",
                    choices=list(STYLE_PRESETS.keys()),
                    value="None",
                )
                random_seed = gr.Checkbox(label="Randomize seed", value=True)
                seed_input = gr.Number(label="Seed", value=42, precision=0)

                with gr.Accordion("Advanced Settings", open=False):
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps",
                    )
                    guidance_slider = gr.Slider(
                        minimum=1,
                        maximum=15,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    width_slider = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                        label="Width",
                    )
                    height_slider = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                        label="Height",
                    )

                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

            with gr.Column(scale=5, elem_classes=["preview-card"]):
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_info = gr.Markdown("Ready to generate.")
                prompt_preview = gr.Textbox(
                    label="Prompt used",
                    interactive=False,
                    lines=2,
                )

        gr.Examples(
            examples=[
                ["A portrait of a cyberpunk engineer in rain, cinematic frame", "Cinematic"],
                ["A magical forest with glowing rivers and ancient ruins", "Fantasy Art"],
                ["A detailed anime mecha standing on a rooftop at sunset", "Anime"],
            ],
            inputs=[prompt_input, preset_dropdown],
        )

        generate_btn.click(
            fn=generate,
            inputs=[
                prompt_input,
                preset_dropdown,
                random_seed,
                seed_input,
                steps_slider,
                guidance_slider,
                width_slider,
                height_slider,
            ],
            outputs=[output_image, generation_info, prompt_preview],
            show_progress="full",
        )

# Launch with local link
demo.launch(share=False)