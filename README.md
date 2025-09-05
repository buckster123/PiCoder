# PiCoder 🚀: Grok-Powered Coding Sidekick on a Pi – Because Who Needs a Supercomputer?

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![xAI Grok](https://img.shields.io/badge/Powered%20by-Grok%20API-orange.svg)](https://x.ai/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Compatible-red.svg?logo=raspberrypi)](https://www.raspberrypi.com/)

> "Don't Panic! PiCoder is here to turn your Raspberry Pi into a coding wizard's lair. Inspired by the infinite improbability drive, but way more probable on a budget." – Grok, probably.

## 📜 Overview

PiCoder is your ultimate low-cost, high-IQ coding platform: a fusion of an advanced AI agent (powered by xAI's Grok API) and a slick Streamlit chat app. Optimized for Raspberry Pi 5 (or clones like Orange Pi), it democratizes AI-assisted coding on affordable hardware. Think of it as Grok trapped in a tiny silicon box, ready to spit out deployable code, debug your scripts, and teach you tricks – all while sipping power like a caffeinated electron.

The AI (PiCoder) is a nerdy genius: creative, self-verifying, and educational, defaulting to Python wizardry but flexing across languages. The app? A neon-lit chat den with tools, auth, and vision superpowers. Cross-platform vibes mean it runs on your Pi, laptop, or that dusty server in the closet. Perfect for makers, hackers, and code ninjas chasing the 42 of programming.

**Why PiCoder?** Because overkill workstations are so 2023. Build bots, web apps, or Linux hacks on a $35 board. [Check out xAI for more Grok lore](https://x.ai/grok).

## ✨ Features

### 🤖 AI Capabilities: PiCoder, the Code Whisperer
- **Epic Code Gen**: Crafts clean, PEP8-compliant Python (default), JS, PHP, Rust – you name it. Meaningful vars, comments, error-handling? Check. [PEP 8 Guide](https://peps.python.org/pep-0008/).
- **Creative Hacks**: Thinks outside the matrix – functional paradigms, pros/cons debates, extensible designs. "Why loop when you can map?"
- **Self-Check Mode**: Inline edge cases, pytest/Jest tests, dynamic runs via tools. Bugs? PiCoder squashes 'em like pixels in a retro game.
- **Nerdy Teaching**: Breaks down concepts with analogies (e.g., "Recursion is like a Russian doll of functions"). Suggests reads like [Python Docs](https://docs.python.org/3/) or [MDN Web Docs](https://developer.mozilla.org/).
- **Agentic Superpowers**: Plans tasks, batches tools, remembers your prefs via hierarchical memory. No more "forgetting" that Rust crate you love.
- **Safety First**: Ethical guardrails – truthful answers, no shady stuff. Dark themes? Optional, but we got violent fictional tales if you're into that (ethically).

### 🖥️ App Features: Streamlit-Powered Command Center
- **Secure Login**: SHA-256 hashed creds with passlib. No "password123" allowed, hacker.
- **Chat Magic**: Streaming replies, history search, auto-titling. Bubbles, expanders, dark mode toggle for those late-night sessions.
- **Customization Station**: Model picker (Grok-4 ftw), editable prompts from files. Upload images for vision quests.
- **UI Flair**: Neon gradients, wrapped code blocks (no scroll-of-doom), avatars. Toggle dark mode for vampire coders.
- **Database Backend**: SQLite with WAL for users/history/memory. Concurrent? Handled like a boss.

### 🔧 Supported Tools: Sandboxed Arsenal for Agentic Wins
All locked in `./sandbox/` – because escaping the matrix is for movies. Invoke via XML calls; PiCoder plans batches to avoid loop hell.

- 📂 **File Ops**: Read/write/list/mkdir (fs_* tools). Stash your scripts here.
- ⏰ **Time Lord**: get_current_time – NTP sync for precision timing attacks (on clocks).
- 🐍 **Code Runner**: code_execution – Stateful Python REPL with numpy/sympy/pygame. Test that algo live!
- 🧠 **Memory Vault**: memory_insert/query – Hierarchical JSON for prefs/projects. Nerd level: Over 9000.
- 📚 **Git Guru**: git_ops – Init/commit/branch/diff. Version your sandbox masterpieces.
- 🗄️ **DB Dabbler**: db_query – SQLite ops in sandbox. Prototype that app DB.
- 🐚 **Shell Shenanigans**: shell_exec – Safe commands (ls/grep/sed). Linux fu without sudo risks.
- ✨ **Lint Wizard**: code_lint – Black-formats Python. Messy code? Not on our watch.
- 🌐 **API Mockery**: api_simulate – Fake or real calls to public APIs. Test without rate limits.

[Streamlit Tool Docs](https://docs.streamlit.io/library/api-reference/utilities/st.experimental_get_query_params) for inspo.

## 🛠️ Use Case Examples: Level Up Your Pi Projects

### 1. **Web Scraper Bot** 🔍
   - Query: "Build a Python scraper for GitHub repos using requests and BeautifulSoup."
   - PiCoder: Plans (batch tools), codes, lints with code_lint, tests via code_execution. Explains selectors like "HTML tags are like treasure maps."
   - Nerd Edge: "Scraping ethically – don't be that bot that DDoS's the galaxy." Output: Deployable script in sandbox.

### 2. **Git Repo Manager** 📂
   - Enable tools, ask: "Init a repo, commit changes, and diff."
   - PiCoder: Uses git_ops in batches. Remembers your branch prefs via memory_query.
   - Nerd Edge: "Git good or git rekt. PiCoder handles the commits so you can focus on the memes."

### 3. **Database Prototype** 🗃️
   - "Create a SQLite DB for user logs and query it."
   - PiCoder: db_query for setup/inserts, shell_exec to ls results. Educational: "SQL joins are like Voltron – better together."
   - Nerd Edge: "From zero to DB hero in one chat. Bonus: Simulate APIs with api_simulate for that full-stack feel."

### 4. **Linux Automation Script** 🐧
   - "Write a Bash script to monitor Pi temps, lint my Python helper."
   - PiCoder: Defaults to Bash, uses shell_exec for demos, code_lint for Python parts.
   - Nerd Edge: "Overheat your Pi? Not with this. It's cooler than absolute zero."

More ideas? [Raspberry Pi Projects](https://projects.raspberrypi.org/) + PiCoder = Infinite Awesomeness.

## 🛡️ Installation: Noob-Proof Guide for Headless Raspberry Pi 5

Designed for headless setup (no monitor needed – SSH in!), but runs great on desktop Raspberry Pi OS too. We'll cover everything step-by-step. If you're a total beginner, follow along; we've got your back.

### Step 0: Prep Your Pi
- Flash Raspberry Pi OS (Lite for headless, or Full for desktop) using [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
- Enable SSH: In Imager, set hostname, enable SSH, set user/password.
- Boot the Pi, find its IP (e.g., via router or `hostname -I`), SSH in: `ssh pi@your-pi-ip` (default user: pi, password: raspberry – change it!).
- Update system: `sudo apt update && sudo apt upgrade -y`.

### Step 1: Install System Dependencies
These are essential for the app and tools. Run the provided `install.sh` script (detailed below), or manually:

- **Core Packages**:
  - `python3` and `python3-venv`: For running the app.
  - `python3-pip`: Package manager.
  - `git`: For cloning and git_ops tool.
  - `sqlite3` and `libsqlite3-dev`: For databases.
  - `libgit2-dev`: For pygit2 (Git tool).
  - `build-essential`: Compilers for building packages.
  - **For code_execution Tool**: To support libraries like numpy/sympy/pygame, we need system deps for compilation (e.g., libatlas-base-dev for numpy, libjpeg-dev for pygame if using images).
    - Note: If you hit "snappy" issues (e.g., python-snappy dep errors), it's likely from optional compression libs – we've avoided it here. For numpy: `sudo apt install libatlas-base-dev`. For pygame: `sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev`. Sympy is pure Python, no extra deps.

Manual install command:
```
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git sqlite3 libsqlite3-dev libgit2-dev build-essential libatlas-base-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev
```

### Step 2: Clone and Set Up
- Clone: `git clone https://github.com/yourusername/picoder.git && cd picoder`
- Venv: `python3 -m venv venv && source venv/bin/activate`
- Install Python packages: `pip install -r requirements.txt`
  - This includes extras for code_execution: numpy, sympy, pygame (pre-compiled wheels work on Pi ARM).

### Step 3: Configure
- Create `.env`: `echo "XAI_API_KEY=your_xai_api_key_here" > .env` (get key from [x.ai](https://x.ai/)).
- (Optional) For headless: Run Streamlit on port 8501, access via browser on another machine: http://your-pi-ip:8501.

### Step 4: Run
- `streamlit run app.py`
- For background/headless: Use `nohup streamlit run app.py &` or systemd service (see below).

### Install Script: install.sh
Save this as `install.sh`, make executable (`chmod +x install.sh`), and run `sudo ./install.sh`. It handles everything for headless Pi 5.

```bash
#!/bin/bash

# PiCoder Install Script for Headless Raspberry Pi 5
# Run as sudo for system packages.

echo "Updating system..."
apt update && apt upgrade -y

echo "Installing system dependencies..."
apt install -y python3 python3-venv python3-pip git sqlite3 libsqlite3-dev libgit2-dev build-essential libatlas-base-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev

echo "Cloning repo..."
git clone https://github.com/yourusername/picoder.git
cd picoder

echo "Setting up venv..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python packages..."
pip install -r requirements.txt

echo "Reminder: Add your XAI_API_KEY to .env file!"
echo "To run: source venv/bin/activate && streamlit run app.py"

# Optional: Systemd service for headless auto-start
cat << EOF > /etc/systemd/system/picoder.service
[Unit]
Description=PiCoder Streamlit App
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/picoder
ExecStart=/home/pi/picoder/venv/bin/streamlit run app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
echo "To enable auto-start: sudo systemctl enable picoder.service"
echo "Installation complete! 🚀"
```

Trouble? Common fixes:
- **Dep Errors**: If numpy fails (e.g., BLAS issues), ensure libatlas-base-dev is installed. For pygame, check SDL deps.
- **Headless Access**: Forward port 8501 or use ngrok ([Ngrok Guide](https://ngrok.com/docs/getting-started/)).
- **Desktop Mode**: Install Raspberry Pi OS with desktop, run in terminal – GUI optional.
- Questions? [Raspberry Pi Forums](https://forums.raspberrypi.com/) or open an issue.

## 🔌 System Package Dependencies (Detailed)

- `python3`, `python3-venv`, `python3-pip`: Core Python.
- `git`: For repo cloning and git_ops.
- `sqlite3`, `libsqlite3-dev`: Database support.
- `libgit2-dev`: Git library for pygit2.
- `build-essential`: Compilers (gcc, etc.) for building wheels.
- `libatlas-base-dev`: For numpy linear algebra.
- SDL libs (`libsdl2-dev` etc.): For pygame in code_execution (graphics/audio).
- No "snappy" required – if you see errors, it's likely from unrelated packages; our reqs avoid it.

For other platforms: Adapt with apt/yum/brew. E.g., macOS: `brew install python git sqlite libgit2`.

## 🚀 Usage

Fire up the app, login (or register – pro tip: strong passwords), pick a model/prompt, enable tools, and chat away. Upload pics for vision, toggle dark mode for stealth mode. Save chats, search history, and let PiCoder handle the heavy lifting.

Pro Tip: Start with "Teach me Rust basics while building a CLI tool." Watch the magic unfold.

## ⚙️ Configuration

- **Prompts**: Drop .txt files in `./prompts/`. Edit/save in-app. Love in filename? Gets a <3 flair.
- **Sandbox**: `./sandbox/` for tools. Pre-load files for AI access.
- **Themes**: CSS in app.py – hack away for custom neon.
- **Extend Tools**: Add to TOOLS list in code. [OpenAI SDK Docs](https://platform.openai.com/docs/libraries/python-library) for inspo.

## 🤝 Contributing

Fork, branch, PR! We're building the ultimate Pi AI rig.
- Focus: Pi optimizations, new tools (e.g., GPIO?), bug hunts.
- Style: PEP8 or bust. Test on real hardware.
- Issues: [File here](https://github.com/yourusername/picoder/issues) with repro steps.

Nerd Bonus: Contributions earn virtual high-fives from Grok.

## 📄 License

MIT – Free as in beer (or electrons). See [LICENSE](LICENSE).

## 🙌 Acknowledgments

- xAI for Grok's brainpower 🧠.
- Streamlit team for the UI sorcery ✨.
- Raspberry Pi Foundation for affordable awesomeness 🍓.
- Open-source libs like pygit2, black, and ntplib.

Questions? Hit up issues or [tweet at xAI](https://twitter.com/xai). Code long and prosper! 🖖
