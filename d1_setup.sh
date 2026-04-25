# ── 1. Check Python version (need 3.10+) ──────────────────────────────────────
python --version
# or on some systems:
python3 --version

# ── 2. Create your project folder ─────────────────────────────────────────────
mkdir fde-learning
cd fde-learning

# ── 3. Create a virtual environment ───────────────────────────────────────────
python -m venv venv

# ── 4. Activate it ────────────────────────────────────────────────────────────
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# ── 5. You'll see (venv) prefix in your terminal — that means it's active ─────
# (venv) $

# ── 6. Install the packages we need today ─────────────────────────────────────
pip install requests anthropic python-dotenv

# ── 7. Save your dependencies to a file ───────────────────────────────────────
pip freeze > requirements.txt

# ── 8. Create project structure ───────────────────────────────────────────────
mkdir src
touch src/main.py
touch .env
touch .gitignore

# ── 9. Add to .gitignore (never commit secrets or venv!) ──────────────────────
echo "venv/" >> .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
