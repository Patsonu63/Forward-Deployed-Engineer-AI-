# ══════════════════════════════════════════════════════
# GIT — Complete workflow for FDE projects
# ══════════════════════════════════════════════════════

# ── First time setup (do once) ────────────────────────
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# ── Initialize a new repo ─────────────────────────────
git init
# Creates a hidden .git folder — your project is now tracked

# ── Check what's changed ──────────────────────────────
git status
# Shows: untracked files (red), staged files (green), modified files

# ── Stage files for commit ────────────────────────────
git add src/main.py          # add one file
git add .                    # add ALL changed files
git add src/                 # add entire folder

# ── Commit (save a snapshot) ─────────────────────────
git commit -m "feat: add customer classification script"
# Good commit message format:
# feat: new feature
# fix: bug fix
# docs: documentation
# refactor: code cleanup

# ── View commit history ───────────────────────────────
git log --oneline
# Shows: abc1234 feat: add customer classification script

# ── Connect to GitHub ─────────────────────────────────
# 1. Go to github.com → New repository → name it "fde-learning"
# 2. Copy the repo URL
git remote add origin https://github.com/yourusername/fde-learning.git
git branch -M main
git push -u origin main    # -u sets upstream (first push only)

# ── Regular push after first time ─────────────────────
git push

# ── Pull latest changes (when collaborating) ──────────
git pull

# ── Create a feature branch ───────────────────────────
git checkout -b feature/add-api-client
# Now you're on a new branch — main is untouched

git add .
git commit -m "feat: add REST API client"
git push origin feature/add-api-client
# Then open a Pull Request on GitHub

# ── Merge branch back to main ─────────────────────────
git checkout main
git merge feature/add-api-client

# ── Undo mistakes ─────────────────────────────────────
git restore filename.py         # discard uncommitted changes to a file
git reset HEAD~1                # undo last commit (keeps changes)
git stash                       # temporarily save uncommitted changes
git stash pop                   # restore stashed changes

# ── .gitignore — files Git will NEVER track ───────────
# Your .gitignore should contain:
# venv/
# .env
# __pycache__/
# *.pyc
# .DS_Store
# *.egg-info/
# dist/
# .pytest_cache/
