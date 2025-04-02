# Discord Image Generation Bot

A Discord bot that uses the `/imagine` command to generate images with multiple AI models through the OpenAI API.

## Features

- Slash command `/imagine` to generate images based on text prompts
- Supports multiple image generation models with automatic fallback
- Tries models in order from highest to lowest quality
- Automatically falls back to the next model if one fails (e.g., due to content policy violations)
- Displays generated images directly in Discord with model attribution
- Compatible with custom OpenAI-compatible API backends

## Supported Models

The bot supports the following models in order of preference (best to worst):

1. **DALL-E 3** - Highest quality, most advanced capabilities
2. **Stable Diffusion XL** (stabilityai/stable-diffusion-xl-base-1.0) - High quality general-purpose model
3. **SDXL Lightning** (bytedance/stable-diffusion-xl-lightning) - Fast SDXL variant
4. **Dreamshaper 8** (lykon/dreamshaper-8-lcm) - Creative stylized outputs
5. **DALL-E 2** - Older OpenAI model as final fallback

All models are accessed through the OpenAI API or a compatible API backend.

## Prerequisites

- Python 3.8 or higher
- A Discord account and a Discord server where you have admin permissions
- An OpenAI API key (or access to a compatible API)

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd imgbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to the "Bot" tab and click "Add Bot"
4. Under the "Privileged Gateway Intents" section, enable "Message Content Intent"
5. Copy the bot token by clicking "Reset Token" and then "Copy"

### 4. Invite the Bot to Your Server

1. In the Discord Developer Portal, go to the "OAuth2" tab
2. In the "OAuth2 URL Generator" section, select the following scopes:
   - `bot`
   - `applications.commands`
3. In the "Bot Permissions" section, select:
   - "Send Messages"
   - "Embed Links"
   - "Attach Files"
   - "Use Slash Commands"
4. Copy the generated URL and open it in your browser
5. Select your server and authorize the bot

### 5. Get an OpenAI API Key

1. Go to the [OpenAI API Keys page](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the API key (you won't be able to see it again)

If you're using a custom API backend that's compatible with the OpenAI API, you'll need to get the API key from that provider instead.

### 6. Configure Environment Variables

1. Create a `.env` file based on the `.env.example` template:
   ```bash
   cp .env.example .env
   ```
2. Edit the `.env` file and add your Discord bot token and API key:
   ```
   DISCORD_TOKEN=your_discord_bot_token_here
   OPENAI_API_KEY=your_api_key_here
   ```
3. The bot is pre-configured to use Function Network as the API backend:
   ```
   OPENAI_API_BASE=https://api.function.network
   ```
   This will direct API requests to Function Network's endpoint at `https://api.function.network/v1/images/generation` for image generation.

## Running the Bot

Run the bot with the following command:

```bash
python main.py
```

If everything is set up correctly, you should see a message indicating that the bot has logged in and the commands have been synced.

## Deploying with AWS Lightsail

AWS Lightsail provides a simple and cost-effective way to deploy your Discord bot. Here's how to set it up:

### 1. Create a Lightsail Instance

1. Sign in to the [AWS Management Console](https://aws.amazon.com/console/)
2. Navigate to the Lightsail service
3. Click "Create instance"
4. Choose a platform: Select "Linux/Unix"
5. Select a blueprint: Choose "OS Only" and select "Ubuntu 22.04 LTS"
6. Choose an instance plan: The smallest plan ($3.50/month) should be sufficient for this bot
7. Name your instance (e.g., "discord-image-bot")
8. Click "Create instance"

### 2. Connect to Your Instance

1. Once your instance is running, click on it to open the management page
2. Click on the "Connect" tab and then "Connect using SSH"
3. This will open a browser-based SSH terminal

### 3. Set Up the Environment

Run the following commands to set up your environment:

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install Python and Git
sudo apt install -y python3-pip python3-venv git

# Create a directory for the bot
mkdir -p ~/discord-bot
cd ~/discord-bot

# Clone the repository
git clone https://github.com/yourusername/imgbot.git .

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure the Bot

1. Create and edit the `.env` file:

```bash
nano .env
```

2. Add your configuration (Discord token, OpenAI API key, etc.):

```
DISCORD_TOKEN=your_discord_bot_token_here
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.function.network
```

3. Save and exit (Ctrl+X, then Y, then Enter)

### 5. Set Up a Systemd Service for Auto-start

1. Create a systemd service file:

```bash
sudo nano /etc/systemd/system/discord-bot.service
```

2. Add the following content:

```
[Unit]
Description=Discord Image Generation Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/discord-bot
ExecStart=/home/ubuntu/discord-bot/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

3. Save and exit (Ctrl+X, then Y, then Enter)

4. Enable and start the service:

```bash
sudo systemctl enable discord-bot.service
sudo systemctl start discord-bot.service
```

5. Check the status to make sure it's running:

```bash
sudo systemctl status discord-bot.service
```

### 6. Monitor the Bot Logs

To view the bot's logs:

```bash
sudo journalctl -u discord-bot.service -f
```

Press Ctrl+C to exit the log view.

### 7. Updating the Bot

To update the bot in the future:

```bash
cd ~/discord-bot
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart discord-bot.service
```

## Using the Bot

1. In your Discord server, type `/imagine` followed by your prompt
2. The bot will attempt to generate an image using the highest quality model first
3. If a model fails (e.g., due to content policy violations), the bot will automatically try the next model
4. The generated image will be displayed in an embed with your prompt and the model used

Example:
```
/imagine a futuristic city with flying cars and neon lights
```

## Notes

- The bot tries models in order from highest to lowest quality
- If all models fail, the bot will display the error messages from each attempt
- Each image generation will count towards your API usage
- All images are generated at 1024x1024 pixels for consistency
- The bot is designed to work with both the official OpenAI API and compatible third-party APIs

## Troubleshooting

- If the bot doesn't respond, check that you've correctly set up the environment variables
- If you get an error about permissions, make sure the bot has the necessary permissions in your Discord server
- If you get errors from all models, check that your API key is valid and has sufficient credits
- The bot is configured to use Function Network's API (https://api.function.network) by default
- For content policy violations, the bot will automatically try the next model in the list
