# Publish to GitHub (maintainers)

The repo is ready locally with an initial commit (includes `models/color_match.onnx`).

## One-time: log in to GitHub CLI

```powershell
gh auth login
```

## Create public repo and push

From this folder:

```powershell
cd path\to\MAGICMATCH
git branch -M main
gh repo create MAGICMATCH --public --source=. --remote=origin --push
```

Or create an empty repo on github.com named `MAGICMATCH`, then:

```powershell
git branch -M main
git remote add origin https://github.com/<your-user>/MAGICMATCH.git
git push -u origin main
```

## ComfyUI install URL (after push)

```text
ComfyUI/custom_nodes/
  git clone https://github.com/<your-user>/MAGICMATCH.git
```
