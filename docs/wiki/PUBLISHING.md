# Publishing these pages to the GitHub wiki

Everything in this folder is generated from the user manual and the
shipped decks and notebooks, so edits made here are overwritten. The
wiki is its own git repository, `ion_gym.wiki.git`, which GitHub
creates when the first page is saved through the web interface.

1. On GitHub, open the repository's Wiki tab and save any first page
   (this creates the wiki repository).
2. Clone it next to the main checkout:
   `git clone https://github.com/bhclowers/ion_gym.wiki.git`
3. Copy this folder's contents over it, excluding this file:
   `rsync -a --delete --exclude PUBLISHING.md --exclude .git docs/wiki/ ../ion_gym.wiki/`
   (`--delete` removes wiki pages that are no longer generated, such as a
   renamed chapter. Check `git status` in the wiki clone before
   committing.)
4. In the wiki clone: `git add -A && git commit -m "Update wiki" && git push`.
