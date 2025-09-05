#!/bin/bash

# install.sh: Noob-Friendly Installer for PiCoder on Raspberry Pi 5 (Headless/Default, Desktop Optional)
# Run with: bash install.sh
# Assumes fresh Raspberry Pi OS (Lite for headless; Full for desktop). Handles deps, venv, pip, and launch.
# Nerd Edge: "Automating setups so you can code faster than light... almost."

set -e  # Exit on errors

# Colors for fancy output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 PiCoder Installer: Let's turn your Pi into a coding beast!${NC}"

# Step 0: Detect Environment (Headless vs Desktop)
if [ -z "$DISPLAY" ]; then
    echo -e "${GREEN}Headless mode detected. We'll set up for SSH/remote access.${NC}"
    HEADLESS=true
else
    echo -e "${GREEN}Desktop detected. App will launch in browser.${NC}"
    HEADLESS=false
fi

# Step 1: Update and Install System Packages
echo -e "${YELLOW}Step 1: Updating system and installing dependencies...${NC}"
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git sqlite3 libsqlite3-dev libgit2-dev build-essential libsnappy-dev  # For snappy/compression deps in pandas/etc.

# Optional: For advanced REPL libs (e.g., torch on ARM may need extras)
sudo apt install -y libatlas-base-dev libopenblas-dev libblas-dev liblapack-dev gfortran  # For numpy/scipy/torch builds

echo -e "${GREEN}System packages installed!${NC}"

# Step 2: Clone Repo (if not already)
if [ ! -d "picoder" ]; then
    echo -e "${YELLOW}Step 2: Cloning PiCoder repo...${NC}"
    git clone https://github.com/yourusername/picoder.git
    cd picoder
else
    echo -e "${GREEN}Repo already cloned. Pulling updates...${NC}"
    cd picoder
    git pull
fi

# Step 3: Set Up Virtual Environment
echo -e "${YELLOW}Step 3: Creating venv...${NC}"
python3 -m venv venv
source venv/bin/activate

# Step 4: Install Python Packages with Piwheels for ARM Optimization
echo -e "${YELLOW}Step 4: Installing requirements (this may take a while on Pi)...${NC}"
pip install --extra-index-url https://www.piwheels.org/simple -r requirements.txt || {
    echo -e "${RED}Some installs failed (e.g., torch/rdkit on ARM). Trying without problematic ones...${NC}"
    # Fallback: Install core, skip heavy ones
    pip install --extra-index-url https://www.piwheels.org/simple streamlit openai passlib python-dotenv ntplib pygit2 requests black numpy scipy pandas matplotlib sympy mpmath statsmodels PuLP astropy pygame chess networkx ecdsa tqdm
    echo -e "${YELLOW}Installed core libs. For full REPL, resolve deps manually (e.g., install torch via piwheels or compile).${NC}"
}

# Handle specific dep issues (e.g., snappy for pandas parquet)
pip install python-snappy || echo -e "${YELLOW}python-snappy skipped – optional for some pandas features.${NC}"

echo -e "${GREEN}Python packages installed!${NC}"

# Step 5: Set Up .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Step 5: Creating .env – Enter your xAI API Key:${NC}"
    read -p "XAI_API_KEY: " api_key
    echo "XAI_API_KEY=$api_key" > .env
else
    echo -e "${GREEN}.env exists. Skipping...${NC}"
fi

# Step 6: Launch the App
echo -e "${YELLOW}Step 6: Launching PiCoder...${NC}"
if $HEADLESS; then
    echo -e "${GREEN}Headless: Running in background. Access via http://$(hostname -I | awk '{print $1}'):8501${NC}"
    echo -e "${YELLOW}Pro Tip: Use SSH tunnel or VNC for remote browser.${NC}"
    nohup streamlit run app.py > app.log 2>&1 &
else
    streamlit run app.py
fi

echo -e "${GREEN}🎉 Installation complete! If issues, check app.log or forums.${NC}"
echo -e "${YELLOW}Troubleshooting: For dep errors (e.g., snappy/torch), run 'sudo apt install libsnappy-dev' or use piwheels. Happy coding! 🖖${NC}"
