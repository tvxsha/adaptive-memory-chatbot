# Setting up the GitHub repo

Since there are three of you on the team, here's the smoothest way to get
this onto GitHub with everyone able to push.

## 1. Create the repo on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `adaptive-memory-chatbot` (or whatever you all agree on)
3. Set it to **Private** (recommended while it's coursework) or Public if
   your course wants it public
4. **Do NOT** initialize with a README, .gitignore, or license — you already
   have those locally, and initializing on GitHub too will cause a merge
   conflict on your first push
5. Click **Create repository**

GitHub will show you a page with setup commands — you want the
"push an existing repository" section, but here it is spelled out:

## 2. Push this starter pack

From inside the `adaptive-memory-chatbot/` folder (the one this README is
in):

```bash
git init
git add .
git commit -m "Initial project scaffold"
git branch -M main
git remote add origin https://github.com/<your-username>/adaptive-memory-chatbot.git
git push -u origin main
```

Replace `<your-username>` with whoever's account owns the repo, and the URL
with the exact one GitHub showed you after creating the repo.

If you get a permission/auth error on push, GitHub no longer accepts your
account password over HTTPS — you'll need either:
- A [Personal Access Token](https://github.com/settings/tokens) (use it in
  place of your password when prompted), or
- SSH keys set up (`git remote set-url origin git@github.com:<your-username>/adaptive-memory-chatbot.git`)

## 3. Add your teammates as collaborators

1. On the repo page: **Settings → Collaborators → Add people**
2. Enter Peehu's and Vrishti's GitHub usernames (or yours, if you're not the
   owner) and send the invite
3. They accept via the email/notification GitHub sends

## 4. Everyone else clones it

Once added as a collaborator, teammates run:

```bash
git clone https://github.com/<owner-username>/adaptive-memory-chatbot.git
cd adaptive-memory-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add their own Groq key
```

## 5. A simple branch workflow for three people

To avoid stepping on each other:

```bash
git checkout -b feature/contradiction-detection   # or whatever you're working on
# ... make changes ...
git add .
git commit -m "Describe what you did"
git push -u origin feature/contradiction-detection
```

Then open a Pull Request on GitHub into `main` so the others can see the diff
before it merges. For a small three-person project you don't need anything
fancier than "one branch per feature, PR into main."

## 6. Keep secrets out of the repo

`.env` is already in `.gitignore` — never commit it. If you ever accidentally
commit an API key, rotate it immediately in the Groq console (delete the old
key, generate a new one) rather than just deleting the file, since it's still
visible in git history.

## 7. Recommended repo structure once you're underway

- Keep `evaluation/` notebooks committed with cleared output cells (or use
  `nbstripout`) so diffs stay readable — a notebook with all outputs baked in
  creates enormous, unreviewable diffs on every commit.
- Add a `results/` folder (gitignored for large files, but commit your final
  charts as `.png` for the report) once you start running real evaluations.
