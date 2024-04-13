import os
import click
import requests
from io import BytesIO
import torch
import logging
from datetime import datetime
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
import ollama
import warnings

# Setup logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Suppress specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(
    action="ignore", category=UserWarning
)

# Environment variable to potentially suppress tqdm bars
os.environ["TQDM_DISABLE"] = "True"

# Constants for prompts
IMAGE_ANALYSIS_PROMPT = "Identify and list all visible food items in the image provided. Specify each item's name, quantity, and measurement units clearly, avoiding any unnecessary commentary."
RECIPE_SUGGESTION_PROMPT = "Generate a concise, detailed recipe based on the provided list of ingredients ({ingredients}). The recipe should be nutritionally balanced and suitable for a gourmet meal, focusing solely on ingredient use and preparation steps without general dietary advice."
RECIPE_ADJUSTMENT_PROMPT = "Adjust the given recipe based solely on the feedback: {feedback_history}. Use the current ingredients: {ingredients}. Provide specific changes to the recipe steps or ingredients, without including general health or nutritional guidance."
VISUAL_DESCRIPTION_PROMPT = "Provide a concise, detailed visual description for a dish made with these ingredients: {ingredients}. Focus strictly on the appearance of the prepared dish, including texture and color details, without any general cooking or serving suggestions."


def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def setup_model():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
        "mps", torch.float16
    )
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="mps"))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, unet=unet, torch_dtype=torch.float16, variant="fp16"
    ).to("mps")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    return pipe


def generate_image(pipe, description):
    image = pipe(description, num_inference_steps=4, guidance_scale=0).images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    valid_description = "".join(
        char for char in description if char.isalnum() or char in [" ", "_"]
    ).replace(" ", "_")
    if len(valid_description) > 50:
        valid_description = valid_description[:50]
    output_path = f"generated_image_{valid_description}_{timestamp}.png"
    image.save(output_path)
    logging.info(f"Image saved to {output_path}")
    return output_path


def load_image(image_path):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        return BytesIO(response.content)
    else:
        with open(image_path, "rb") as file:
            return file.read()


def analyze_image_for_food(image_data):
    response = ollama.chat(
        model="llava",
        messages=[
            {"role": "user", "content": IMAGE_ANALYSIS_PROMPT, "images": [image_data]}
        ],
    )
    return response["message"]["content"]


def suggest_recipe(ingredients, feedback_history):
    prompt = RECIPE_ADJUSTMENT_PROMPT if feedback_history else RECIPE_SUGGESTION_PROMPT
    feedback_text = "; ".join(feedback_history) if feedback_history else ""
    formatted_prompt = prompt.format(
        feedback_history=feedback_text, ingredients=ingredients
    )
    response = ollama.chat(
        model="llama2", messages=[{"role": "user", "content": formatted_prompt}]
    )
    return response["message"]["content"]


def generate_visual_description(ingredients):
    prompt = VISUAL_DESCRIPTION_PROMPT.format(ingredients=ingredients)
    response = ollama.chat(
        model="llama2", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def write_markdown_file(recipe, image_path):
    markdown_content = f"""
# Final Recipe Presentation

## Recipe Details
{recipe}

## Recipe Image
![Final Dish](./{image_path})

**Enjoy your meal!**
"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    readme_filename = f"recipe_{timestamp}.md"
    with open(readme_filename, "w") as md_file:
        md_file.write(markdown_content)
    return readme_filename


@click.command()
@click.argument("image_path", type=str)
def main(image_path):
    clear_screen()  # Clear the screen before starting the process
    print(f"Loading image: {image_path}")
    image_data = load_image(image_path)
    print(f"Extracting ingredients")
    ingredients = analyze_image_for_food(image_data)
    clear_screen()
    print(f"Identified ingredients {ingredients}")
    feedback_history = []

    recipe = suggest_recipe(ingredients, feedback_history)
    print(f"Building recipe")
    clear_screen()
    print("Initial Recipe Suggestion:\n" + recipe)

    while True:
        feedback = input(
            "Enter your feedback (type 'done' if you are satisfied with the recipe): "
        )
        clear_screen()
        print(f"Applying feedback: {feedback}")
        if feedback.lower() == "done":
            clear_screen()
            print("Generating Recipe Document")
            print("Final Recipe:\n" + recipe)
            visual_description = generate_visual_description(recipe)
            print("Visual Description for Image Generation:\n" + visual_description)
            pipe = setup_model()
            image_path = generate_image(pipe, visual_description)
            file_path = write_markdown_file(recipe, image_path)
            print(f"Find your recipe here: {file_path}")
            break
        else:
            feedback_history.append(feedback)
            clear_screen()  # Clear the screen for each iteration
            recipe = suggest_recipe(ingredients, feedback_history)
            print("Updated Recipe Based on Feedback:\n" + recipe)


if __name__ == "__main__":
    main()
