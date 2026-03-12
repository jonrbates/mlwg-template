# mlwg-template

## Session 1 setup

1. Install vs code w/python extension
2. Install python3.11 with venv
3. Setup git
4. git clone the repo
5. Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install --upgrade pip

# install pytorch
pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cpu

pip install -e .
pip install ".[dev]"
```

## Move to your own repo

```bash
# confirm current origin
git remote -v

# make your own repo (e.g. tickets) and then
git remote remove origin
git remote add origin https://github.com/<username>/<your-repo-name>.git
git push -u origin main
```
