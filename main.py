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
                {"role": "system", "content": "You are an expert at crafting detailed, vivid prompts for image generation. Your task is to enhance the user's prompt to create a more detailed and visually appealing image. Add details about lighting, style, mood, and composition. Keep the essence of the original prompt but make it more descriptive. Respond ONLY with the enhanced prompt, nothing else. If the prompt contains inappropriate content, respond with 'CONTENT_POLICY_VIOLATION' and nothing else."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        enhanced_prompt = response.choices[0].message.content.strip()
        
        # Verify that we got a valid response
        if not enhanced_prompt:
            print("Beautification returned empty result, using original prompt")
            return prompt
        
        # Check if the response indicates a content policy violation or refusal
        refusal_phrases = [
            "i cannot", "i'm unable to", "i am unable to", 
            "i can't", "cannot create", "can't create",
            "unable to create", "not appropriate", "inappropriate",
            "against policy", "content policy", "CONTENT_POLICY_VIOLATION",
            "violates", "violation", "not allowed", "prohibited"
        ]
        
        for phrase in refusal_phrases:
            if phrase.lower() in enhanced_prompt.lower():
                print(f"Content policy violation detected: {enhanced_prompt}")
                # Return a special marker that will be caught by the image generation function
                return "CONTENT_POLICY_VIOLATION"
            
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

import aiohttp
from io import BytesIO
from PIL import Image
import numpy as np

async def is_black_image(image_url, threshold=0.95):
    """
    Check if an image is mostly black (which can happen with content filter blocks).
    
    Args:
        image_url: URL of the image to check
        threshold: Percentage of pixels that need to be black for the image to be considered "black"
        
    Returns:
        True if the image is mostly black, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    return False
                
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))
                
                # Convert to grayscale and get pixel data
                gray_image = image.convert('L')
                pixels = np.array(gray_image)
                
                # Count dark pixels (value < 10 on a 0-255 scale)
                dark_pixels = np.sum(pixels < 10)
                total_pixels = pixels.size
                
                # Calculate the ratio of dark pixels
                dark_ratio = dark_pixels / total_pixels
                
                print(f"Image darkness check: {dark_ratio:.2f} of pixels are dark (threshold: {threshold})")
                
                # Return True if the ratio exceeds the threshold
                return dark_ratio > threshold
    except Exception as e:
        print(f"Error checking if image is black: {str(e)}")
        return False  # Assume it's not black if we can't check

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
        
        # Check if the image is mostly black (content filter)
        if await is_black_image(image_url):
            return None, "Generated image appears to be blocked by content filter (black image)"
        
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
        await interaction.response.defer(thinking=True, ephemeral=True)
        
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
        
        # Check if the prompt was flagged as inappropriate by the beautification process
        if beautified_prompt == "CONTENT_POLICY_VIOLATION":
            print(f"Content policy violation detected in prompt beautification: {updated_prompt}")
            # Send a warning to the user (ephemeral)
            await interaction.followup.send(
                "Warning: Your prompt may contain inappropriate content. I'll try to generate an image using the original prompt, but some models may reject it.",
                ephemeral=True
            )
            # Use the updated prompt with descriptions instead of the beautified one
            beautified_prompt = updated_prompt
        
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
            # Use channel.send instead of followup.send to ensure the message is visible to everyone
            await interaction.channel.send(embed=embed)
            
            # Send a confirmation to the user (ephemeral)
            await interaction.followup.send("Image generated successfully!", ephemeral=True)
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
        await interaction.response.defer(thinking=True, ephemeral=True)
        
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
        
        # Check if the prompt was flagged as inappropriate by the beautification process
        if beautified_prompt == "CONTENT_POLICY_VIOLATION":
            print(f"Content policy violation detected in prompt beautification: {updated_prompt}")
            # Send a warning to the user (ephemeral)
            await interaction.followup.send(
                "Warning: Your prompt may contain inappropriate content. I'll try to generate an image using the original prompt, but some models may reject it.",
                ephemeral=True
            )
            # Use the updated prompt with descriptions instead of the beautified one
            beautified_prompt = updated_prompt
        
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
            # Use channel.send instead of followup.send to ensure the message is visible to everyone
            await interaction.channel.send(embed=embed)
            
            # Send a confirmation to the user (ephemeral)
            await interaction.followup.send("Image generated successfully!", ephemeral=True)
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
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        # Beautify the original prompt without adding descriptions
        beautified_prompt = await beautify_prompt(self.original_prompt)
        
        # Check if the prompt was flagged as inappropriate by the beautification process
        if beautified_prompt == "CONTENT_POLICY_VIOLATION":
            print(f"Content policy violation detected in prompt beautification: {self.original_prompt}")
            # Send a warning to the user (ephemeral)
            await interaction.followup.send(
                "Warning: Your prompt may contain inappropriate content. I'll try to generate an image using the original prompt, but some models may reject it.",
                ephemeral=True
            )
            # Use the original prompt instead of the beautified one
            beautified_prompt = self.original_prompt
        
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
            # Use channel.send instead of followup.send to ensure the message is visible to everyone
            await interaction.channel.send(embed=embed)
            
            # Send a confirmation to the user (ephemeral)
            await interaction.followup.send("Image generated successfully!", ephemeral=True)
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
    # Use ephemeral=True for the initial defer to make the "thinking..." message private
    await interaction.response.defer(thinking=True, ephemeral=True)
    
    try:
        # Detect names in the prompt
        names = await detect_names(prompt)
        
        if names and len(names) > 0:
            # Names detected, ask for descriptions
            view = NameOptionsView(names, prompt)
            # Make sure to use ephemeral=True to make this message visible only to the requester
            await interaction.followup.send(
                f"I detected the following names in your prompt: {', '.join(names)}. What would you like to do?",
                view=view,
                ephemeral=True
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
        
        # Check if the prompt was flagged as inappropriate by the beautification process
        if beautified_prompt == "CONTENT_POLICY_VIOLATION":
            print(f"Content policy violation detected in prompt beautification: {prompt}")
            # Send a warning to the user (ephemeral)
            await interaction.followup.send(
                "Warning: Your prompt may contain inappropriate content. I'll try to generate an image using the original prompt, but some models may reject it.",
                ephemeral=True
            )
            # Use the original prompt instead of the beautified one
            beautified_prompt = prompt
        
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
            # Use channel.send instead of followup.send to ensure the message is visible to everyone
            await interaction.channel.send(embed=embed)
            
            # Send a confirmation to the user (ephemeral)
            await interaction.followup.send("Image generated successfully!", ephemeral=True)
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

@bot.tree.command(name="imaginequote", description="Generate an image based on a random quote from the server")
async def imaginequote(interaction: discord.Interaction):
    """
    Slash command to generate an image based on a random quote from the server.
    
    Args:
        interaction: The Discord interaction object
    """
    # Use ephemeral=True for the initial defer to make the "thinking..." message private
    await interaction.response.defer(thinking=True, ephemeral=True)
    
    try:
        # Send a message to trigger UB3R-B0T to post a quote
        quote_request_msg = await interaction.channel.send(".quote")
        
        # Wait for UB3R-B0T to respond with a quote
        def check_quote_response(message):
            # Check if the message is from UB3R-B0T and contains a quote
            return (
                message.author.name == "UB3R-B0T" and 
                message.channel.id == interaction.channel.id and
                "#" in message.content
            )
        
        try:
            # Wait for UB3R-B0T's response (timeout after 10 seconds)
            quote_response = await bot.wait_for('message', check=check_quote_response, timeout=10.0)
            
            # Extract the quote text (between the quote number and the attribution)
            quote_content = quote_response.content
            
            # Find the position after the quote number (e.g., "#1274")
            quote_start_match = re.search(r'#\d+\s*', quote_content)
            if not quote_start_match:
                await interaction.followup.send("Couldn't parse the quote format.", ephemeral=True)
                return
            
            quote_start = quote_start_match.end()
            
            # Find the position of the attribution (starts with @)
            attribution_match = re.search(r'@[\w\s]+\(', quote_content[quote_start:])
            if attribution_match:
                quote_end = quote_start + attribution_match.start()
            else:
                # If no attribution found, use the rest of the content
                quote_end = len(quote_content)
            
            # Extract the quote text
            quote_text = quote_content[quote_start:quote_end].strip()
            
            # If we got a valid quote, generate an image based on it
            if quote_text:
                await generate_and_send_image(interaction, quote_text)
            else:
                await interaction.followup.send("Couldn't extract a valid quote from the response.", ephemeral=True)
                
        except asyncio.TimeoutError:
            # If UB3R-B0T doesn't respond in time
            await interaction.followup.send("Timed out waiting for a quote response from UB3R-B0T.", ephemeral=True)
            
    except Exception as e:
        # If any part of the process fails
        print(f"Error in imaginequote command: {str(e)}")
        await interaction.followup.send(f"An error occurred: {str(e)}", ephemeral=True)

# Run the bot
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables")
    elif not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
    else:
        bot.run(DISCORD_TOKEN)
