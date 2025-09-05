#!/bin/bash

# install.sh: Noob-Friendly Installer for PiCoder on Raspberry Pi 5 (Headless Default)
# Usage: ./install.sh your_xai_api_key_here
# Assumes fresh Raspberry Pi OS Lite. Run as pi user with sudo access.

API_KEY="$1"
if [ -z "$API_KEY" ]; then
  echo "Error: Provide your XAI_API_KEY as argument. E.g., ./install.sh yourkey"
  exit 1
fi

echo "🚀 Starting PiCoder Install - Hold onto your electrons!"

# Step 1: Update System & Install Dependencies
echo "📦 Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git sqlite3 libsqlite3-dev libgit2-dev build-essential libatlas-base-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev pkg-config libjansson-dev libavcodec-dev libavformat-dev libswscale-dev libjpeg-dev libtiff5-dev libwebp-dev

# Optional: Install Desktop if wanted (comment out if headless only)
# echo "🖥️ Installing desktop? (y/n)"
# read -p "Choice: " install_desktop
# if [ "$install_desktop" = "y" ]; then
#   sudo apt install -y raspberrypi-ui-mods xserver-xorg lightdm
#   sudo raspi-config nonint do_boot_behaviour B4  # Boot to desktop
# fi

# Step 2: Clone Repo (if not exists)
if [ ! -d "picoder" ]; then
  echo "📥 Cloning repo..."
  git clone https://github.com/yourusername/picoder.git
fi
cd picoder || exit

# Step 3: Create Virtual Env & Install Pip Packages
echo "🐍 Setting up venv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install streamlit openai python-dotenv passlib ntplib pygit2 requests black numpy sympy pygame
# Handle potential dep issues (e.g., snappy if needed; not core but fallback)
pip install python-snappy || echo "Snappy optional - skipping."

# Step 4: Set Up .env
echo "🔑 Configuring .env..."
echo "XAI_API_KEY=$API_KEY" > .env

# Step 5: Create Directories
echo "📂 Creating prompts and sandbox..."
mkdir -p prompts sandbox

# Step 6: Test Imports (for code_execution)
echo "🧪 Testing key imports..."
python -c "import numpy; import sympy; import pygame; print('All good!')"

# Done!
echo "🎉 Install complete! Run with: source venv/bin/activate && streamlit run app.py --server.headless true --server.port 8501"
echo "Access at http://$(hostname -I | awk '{print $1}'):8501 from your network."
echo "For desktop: Just run 'streamlit run app.py' and open in browser."
