# PiCoder: Grok-Powered Coding Agent Sidekick on a Pi – Because Who Needs a Supercomputer?

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![xAI Grok](https://img.shields.io/badge/Powered%20by-Grok%20API-orange.svg)](https://x.ai/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Compatible-red.svg?logo=raspberrypi)](https://www.raspberrypi.com/)

![PiCoder Banner](https://github.com/buckster123/PiCoder/blob/main/banner.jpg)

![PiCoder Banner](https://your-image-url-or-repo-path.png)

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
- 🐍 **Code Runner**: code_execution – Stateful Python REPL with numpy/sympy. Test that algo live!
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

## 🛡️ Installation

### Prerequisites
- **Hardware**: Raspberry Pi 5 💻 (or clones). Desktop? Sure, but where's the fun?
- **Software**:
  - Python 3.8+ 🐍 ([Download](https://www.python.org/downloads/)).
  - Git 📚 ([Install Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)).
  - SQLite 🔗 (usually pre-installed; [Docs](https://www.sqlite.org/docs.html)).
  - libgit2 for pygit2 ([Debian Guide](https://packages.debian.org/search?keywords=libgit2-dev)).
- **API Key**: Grab from [xAI Dashboard](https://x.ai/). Store in `.env`.

### Steps
1. Clone: `git clone https://github.com/yourusername/picoder.git && cd picoder`
2. Venv: `python3 -m venv venv && source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Env: Echo `XAI_API_KEY=yourkey` > .env
5. Launch: `streamlit run app.py`
6. Browse: http://localhost:8501 (or Pi IP for remote hacking).

Trouble? [Streamlit Forum](https://discuss.streamlit.io/) or open an issue.

## 🔌 System Package Dependencies

For Raspberry Pi OS (Debian-based):
```
sudo apt update
sudo apt install python3 python3-venv python3-pip git sqlite3 libsqlite3-dev libgit2-dev build-essential
```
- **Why?** Python for runtime, Git for ops, SQLite for DB, libgit2 for pygit2 magic.
- Cross-platform notes: On Windows, use WSL; macOS via Homebrew ([Brew Git](https://formulae.brew.sh/formula/git)).

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
