# How to put this on GitHub

Written for someone who has never used GitHub. No commands to memorise —
everything is done in the browser by dragging files.

---

Your name (Matin Havaei) is already filled into `LICENSE`, `CITATION.cff` and
`README.md`. Nothing to edit before uploading.

---

## Step 1 — Make a GitHub account

Go to **github.com** and click **Sign up**. Use your university email if you
have one — it looks better on a CV and gets you free student benefits.

Pick a username you won't be embarrassed by in five years. Your real name, or
something close to it, is the safe choice.

---

## Step 2 — Create an empty repository

1. Once logged in, click the **`+`** in the top-right corner
2. Choose **New repository**
3. Fill in:
   - **Repository name:** `pet-frame-prediction`
   - **Description:** `Benchmark of 14 deep learning models for temporal frame prediction in simulated dynamic PET`
   - Select **Public**
   - **Do NOT tick** "Add a README file"
   - **Do NOT tick** "Add .gitignore"
   - **Do NOT** choose a license
4. Click **Create repository**

> Those three boxes are left unticked because your folder already contains a
> README, a .gitignore and a LICENSE. Ticking them creates duplicates that you'd
> then have to untangle.

You'll land on a mostly empty page with setup instructions. Ignore all of it.

---

## Step 3 — Upload the files

1. On that page, find the link **uploading an existing file** (it's in the
   middle of the text). Click it.
2. Open the `repo` folder on your computer in File Explorer
3. Select **everything inside** it — press `Ctrl + A`
4. Drag it all onto the GitHub page

> **Important:** drag the *contents* of the folder, not the folder itself.
> If you drag the folder you'll end up with `repo/repo/README.md`, which looks wrong.

Wait for all files to finish uploading. There are 47 of them, so give it a
minute.

4. In the box at the bottom labelled **Commit changes**, type:
   ```
   Initial commit: models, pipeline and results
   ```
5. Click the green **Commit changes** button

Done. Refresh the page and your README will be displayed with the results table
and figures.

---

## Step 4 — Make it look professional (2 minutes)

On your repository's main page, click the **gear icon** next to "About" on the
right side, then:

- **Description:** `Benchmark of 14 deep learning models for temporal frame prediction in simulated dynamic PET. Leave-one-tumour-out cross-validation.`
- **Topics** (type each, press Enter):
  `deep-learning` `medical-imaging` `pet-imaging` `pytorch` `diffusion-models`
  `variational-autoencoder` `gan` `cross-validation` `time-series-prediction`
- Tick **Releases** and **Packages** off if you like — they're unused

Click **Save changes**.

---

## Step 5 — Check the checklist

Look at your repository page and confirm:

- [ ] The README shows the results table and the ranking figure
- [ ] There is **no** `IMAGES` folder (your data should not be uploaded)
- [ ] There is **no** `.venv` folder
- [ ] There are **no** `.pt` or `.pth` files (model weights are large)
- [ ] The figures in `figures/` display when you click them

If a data or venv folder did get uploaded, see "Removing something" below.

---

## Making changes later

### Editing one file
Click the file → click the **pencil icon** → edit → scroll down →
**Commit changes**.

### Adding more files
On the main page: **Add file** → **Upload files** → drag → **Commit changes**.

### Removing something
Click the file → click the **trash icon** → **Commit changes**.

To delete a whole folder you have to delete the files inside it one at a time
(GitHub's web interface has no folder delete). Annoying, but rare.

---

## Adding the link to your CV

Once uploaded, your repository lives at:

```
https://github.com/<your-username>/pet-frame-prediction
```

On a CV, write it like this:

> **Temporal Frame Prediction in Simulated Dynamic PET** — benchmarked 14 deep
> learning architectures (diffusion, VAE, GAN, transformer) under
> leave-one-tumour-out cross-validation; showed that implementation choices
> outweighed architectural choice (+0.518 vs +0.11 SSIM across architectures).
> `github.com/<your-username>/pet-frame-prediction`

Replace `<your-username>` with whatever username you picked in Step 1.

---

## Three things people get wrong

**Never upload your dataset.** The `.gitignore` blocks image folders, but
drag-and-drop uploads bypass `.gitignore` entirely. Just don't select the
`IMAGES` folder when you drag.

**Never upload `.venv`.** It's hundreds of megabytes of files anyone can
reinstall in one command.

**Don't upload trained model weights** (`.pt` files) unless someone asks. They're
large and GitHub rejects files over 100 MB.

---

## If you want to learn the proper way later

The drag-and-drop method above is perfectly legitimate and plenty of researchers
use it. But if you end up updating the repo often, installing **GitHub Desktop**
(desktop.github.com) is worth an evening. It gives you a button that syncs a
folder on your PC with GitHub, so you edit files normally and press "Push".

You do not need it for this project.
