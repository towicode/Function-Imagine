import os
import discord
import asyncio
import re
from openai import OpenAI
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get tokens from environment variables
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')  # Optional custom API base URL

# Configure OpenAI API client
if OPENAI_API_BASE:
    # Function Network requires the /v1 path in the base URL
    base_url = OPENAI_API_BASE
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Set up intents for the bot
intents = discord.Intents.default()
intents.message_content = True

# Create bot instance
bot = commands.Bot(command_prefix='!', intents=intents)

# Define models in order from best to worst
# These models should be supported by Function Network
MODELS = [
    {
        "name": "dall-e-3",
        "display_name": "DALL-E 3"
    },
    {
        "name": "stabilityai/stable-diffusion-xl-base-1.0",
        "display_name": "Stable Diffusion XL"
    },
    {
        "name": "bytedance/stable-diffusion-xl-lightning",
        "display_name": "SDXL Lightning"
    },
    {
        "name": "lykon/dreamshaper-8-lcm",
        "display_name": "Dreamshaper 8"
    },
    {
        "name": "dall-e-2",
        "display_name": "DALL-E 2"
    }
]

@bot.event
async def on_ready():
    """Event triggered when the bot is ready and connected to Discord."""
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print('------')
    
    # Sync commands with Discord
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(f'Failed to sync commands: {e}')

async def beautify_prompt(prompt):
    """
    Use AI to enhance the prompt for better image generation.
    
    Args:
        prompt: The original user prompt
        
    Returns:
        Enhanced prompt or original prompt if enhancement fails
    """
    if not prompt or not prompt.strip():
        return prompt  # Return as is if prompt is empty or whitespace
        
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": "You are an expert at crafting detailed, vivid prompts for image generation. Your task is to enhance the user's prompt to create a more detailed and visually appealing image. Add details about lighting, style, mood, and composition. Keep the essence of the original prompt but make it more descriptive. Respond ONLY with the enhanced prompt, nothing else."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        enhanced_prompt = response.choices[0].message.content.strip()
        
        # Verify that we got a valid response
        if not enhanced_prompt:
            print("Beautification returned empty result, using original prompt")
            return prompt
            
        print(f"Original prompt: {prompt}")
        print(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        print(f"Error enhancing prompt: {str(e)}")
        return prompt  # Return original prompt if enhancement fails

async def detect_names(prompt):
    """
    Detect potential names of people or characters in the prompt.
    
    Args:
        prompt: The user prompt
        
    Returns:
        List of detected names or empty list if none found
    """
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": "You are an AI assistant that identifies names of people or characters in text. Extract ONLY proper names that appear to be referring to specific individuals. Respond with a JSON array of strings containing ONLY the names, nothing else. If no names are found, respond with an empty array."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        # Try to parse the result as JSON
        try:
            # Handle cases where the model might wrap the array in code blocks or add explanatory text
            result = result.replace("```json", "").replace("```", "").strip()
            # Find anything that looks like a JSON array
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                result = match.group(0)
            names = eval(result)  # Using eval since the response should be a simple array
            return names if isinstance(names, list) else []
        except:
            # If parsing fails, try a simple regex approach as fallback
            potential_names = re.findall(r'["\'](.*?)["\']', result)
            return potential_names
    except Exception as e:
        print(f"Error detecting names: {str(e)}")
        return []  # Return empty list if detection fails

async def generate_image(model_name, prompt):
    """
    Generate an image using Function Network's API.
    
    Args:
        model_name: The model to use
        prompt: The text prompt to generate an image from
        
    Returns:
        Tuple of (image_url, model_display_name) or (None, error_message)
    """
    try:
        # Call Function Network API to generate image
        response = client.images.generate(
            prompt=prompt,
            model=model_name,
            size="1024x1024",
            n=1,
        )
        
        # Get the image URL from the response
        image_url = response.data[0].url
        
        # Get the display name for the model
        model_display_name = next((m["display_name"] for m in MODELS if m["name"] == model_name), model_name)
        
        return image_url, model_display_name
    
    except Exception as e:
        return None, str(e)

async def try_generate_image(prompt):
    """
    Try to generate an image using multiple models in sequence.
    If one model fails, try the next one.
    
    Args:
        prompt: The text prompt to generate an image from
        
    Returns:
        Tuple of (image_url, model_used, error_messages) where error_messages is a dict
        mapping model names to error messages
    """
    error_messages = {}
    
    for model in MODELS:
        model_name = model["name"]
        
        # Try to generate image with the current model
        image_url, result = await generate_image(model_name, prompt)
        
        # If successful, return the image URL and model used
        if image_url:
            return image_url, model["display_name"], error_messages
        
        # If failed, add error message and try next model
        error_messages[model_name] = result
    
    # If all models failed, return None
    return None, None, error_messages

class DescriptionModal(discord.ui.Modal):
    """Modal for collecting descriptions for names."""
    
    def __init__(self, names, original_prompt):
        super().__init__(title="Describe Characters")
        self.names = names
        self.descriptions = {}
        self.original_prompt = original_prompt
        
        # Add text inputs for each name
        for name in names:
            self.add_item(
                discord.ui.TextInput(
                    label=f"Describe {name}",
                    placeholder=f"Enter a description for {name}...",
                    required=False,
                    custom_id=name
                )
            )
    
    async def on_submit(self, interaction: discord.Interaction):
        """Handle the modal submission."""
        await interaction.response.defer(thinking=True)
        
        # Get descriptions from text inputs
        for child in self.children:
            if isinstance(child, discord.ui.TextInput) and child.value:
                name = child.custom_id
                self.descriptions[name] = child.value
        
        # Process the image with the descriptions
        await self.process_image(interaction)
    
    async def process_image(self, interaction):
        """Process the image generation with the collected descriptions."""
        # Update the prompt with the descriptions
        updated_prompt = self.original_prompt
        for name, description in self.descriptions.items():
            updated_prompt = updated_prompt.replace(name, f"{name} ({description})")
        
        # Beautify the prompt
        beautified_prompt = await beautify_prompt(updated_prompt)
        
        # Generate the image
        image_url, model_used, error_messages = await try_generate_image(beautified_prompt)
        
        if image_url:
            # Create an embed with the generated image
            embed = discord.Embed(
                title="Generated Image",
                description=f"Prompt: {self.original_prompt}",
                color=discord.Color.blue()
            )
            embed.set_image(url=image_url)
            embed.set_footer(text=f"Generated with {model_used}")
            
            # Send the embed with the image - this is the final image message, so it's public
            await interaction.followup.send(embed=embed)
        else:
            # If all models failed, send error message (ephemeral)
            error_msg = "Failed to generate image with all available models:\n\n"
            for model in MODELS:
                model_name = model["display_name"]
                if model["name"] in error_messages:
                    error_msg += f"• {model_name}: {error_messages[model['name']]}\n"
            
            await interaction.followup.send(error_msg, ephemeral=True)

class NameOptionsView(discord.ui.View):
    """View with options for handling detected names."""
    
    def __init__(self, names, original_prompt):
        super().__init__(timeout=300)  # 5 minute timeout
        self.names = names
        self.original_prompt = original_prompt
    
    @discord.ui.button(label="Describe Names", style=discord.ButtonStyle.primary)
    async def describe_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle the 'Describe Names' button click."""
        # Create and send the modal for descriptions
        modal = DescriptionModal(self.names, self.original_prompt)
        await interaction.response.send_modal(modal)
    
    @discord.ui.button(label="Surprise Me", style=discord.ButtonStyle.success)
    async def surprise_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle the 'Surprise Me' button click."""
        await interaction.response.defer(thinking=True)
        
        # Generate descriptions for all names
        descriptions = {}
        for name in self.names:
            # Generate a random description for the name
            try:
                response = await client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[
                        {"role": "system", "content": "You are a creative assistant. Generate a brief, imaginative description for a character or person with the given name. Make it visually descriptive and suitable for image generation. Keep it to 1-2 sentences."},
                        {"role": "user", "content": f"Generate a description for a character named {name}"}
                    ],
                    max_tokens=100
                )
                description = response.choices[0].message.content.strip()
                descriptions[name] = description
            except Exception as e:
                print(f"Error generating description for {name}: {str(e)}")
                descriptions[name] = f"a person named {name}"
        
        # Update the prompt with the descriptions
        updated_prompt = self.original_prompt
        for name, description in descriptions.items():
            updated_prompt = updated_prompt.replace(name, f"{name} ({description})")
        
        # Beautify the prompt
        beautified_prompt = await beautify_prompt(updated_prompt)
        
        # Generate the image
        image_url, model_used, error_messages = await try_generate_image(beautified_prompt)
        
        if image_url:
            # Create an embed with the generated image
            embed = discord.Embed(
                title="Generated Image",
                description=f"Prompt: {self.original_prompt}",
                color=discord.Color.blue()
            )
            embed.set_image(url=image_url)
            embed.set_footer(text=f"Generated with {model_used}")
            
            # Send the embed with the image - this is the final image message, so it's public
            await interaction.followup.send(embed=embed)
        else:
            # If all models failed, send error message (ephemeral)
            error_msg = "Failed to generate image with all available models:\n\n"
            for model in MODELS:
                model_name = model["display_name"]
                if model["name"] in error_messages:
                    error_msg += f"• {model_name}: {error_messages[model['name']]}\n"
            
            await interaction.followup.send(error_msg, ephemeral=True)
    
    @discord.ui.button(label="Skip & Generate", style=discord.ButtonStyle.secondary)
    async def skip_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle the 'Skip & Generate' button click."""
        await interaction.response.defer(thinking=True)
        
        # Beautify the original prompt without adding descriptions
        beautified_prompt = await beautify_prompt(self.original_prompt)
        
        # Generate the image
        image_url, model_used, error_messages = await try_generate_image(beautified_prompt)
        
        if image_url:
            # Create an embed with the generated image
            embed = discord.Embed(
                title="Generated Image",
                description=f"Prompt: {self.original_prompt}",
                color=discord.Color.blue()
            )
            embed.set_image(url=image_url)
            embed.set_footer(text=f"Generated with {model_used}")
            
            # Send the embed with the image - this is the final image message, so it's public
            await interaction.followup.send(embed=embed)
        else:
            # If all models failed, send error message (ephemeral)
            error_msg = "Failed to generate image with all available models:\n\n"
            for model in MODELS:
                model_name = model["display_name"]
                if model["name"] in error_messages:
                    error_msg += f"• {model_name}: {error_messages[model['name']]}\n"
            
            await interaction.followup.send(error_msg, ephemeral=True)

@bot.tree.command(name="imagine", description="Generate an image based on your prompt")
async def imagine(interaction: discord.Interaction, prompt: str):
    """
    Slash command to generate an image using multiple AI models with fallback.
    
    Args:
        interaction: The Discord interaction object
        prompt: The text prompt to generate an image from
    """
    await interaction.response.defer(thinking=True)
    
    try:
        # Detect names in the prompt
        names = await detect_names(prompt)
        
        if names and len(names) > 0:
            # Names detected, ask for descriptions
            view = NameOptionsView(names, prompt)
            await interaction.followup.send(
                f"I detected the following names in your prompt: {', '.join(names)}. What would you like to do?",
                view=view,
                ephemeral=True  # Make this message visible only to the requester
            )
        else:
            # No names detected, proceed with prompt beautification and image generation
            await generate_and_send_image(interaction, prompt)
    except Exception as e:
        # If any part of the process fails, still try to generate the image with the original prompt
        print(f"Error in imagine command: {str(e)}")
        await generate_and_send_image(interaction, prompt)

async def generate_and_send_image(interaction, prompt):
    """
    Helper function to generate and send an image.
    
    Args:
        interaction: The Discord interaction object
        prompt: The text prompt to generate an image from
    """
    try:
        # Try to beautify the prompt, but use original if it fails
        beautified_prompt = await beautify_prompt(prompt)
        
        # Try to generate image with multiple models
        image_url, model_used, error_messages = await try_generate_image(beautified_prompt)
        
        if image_url:
            # Create an embed with the generated image
            embed = discord.Embed(
                title="Generated Image",
                description=f"Prompt: {prompt}",
                color=discord.Color.blue()
            )
            embed.set_image(url=image_url)
            embed.set_footer(text=f"Generated with {model_used}")
            
            # Send the embed with the image - this is the final image message, so it's public
            await interaction.followup.send(embed=embed)
        else:
            # If all models failed, send error message (ephemeral)
            error_msg = "Failed to generate image with all available models:\n\n"
            for model in MODELS:
                model_name = model["display_name"]
                if model["name"] in error_messages:
                    error_msg += f"• {model_name}: {error_messages[model['name']]}\n"
            
            await interaction.followup.send(error_msg, ephemeral=True)
    except Exception as e:
        # If something goes wrong, inform the user (ephemeral)
        print(f"Error generating image: {str(e)}")
        await interaction.followup.send(f"An error occurred while generating the image: {str(e)}", ephemeral=True)

# Run the bot
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables")
    elif not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
    else:
        bot.run(DISCORD_TOKEN)
