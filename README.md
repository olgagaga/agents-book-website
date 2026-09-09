# AI Agent Engineering — book website

A simple reading site for the first ten chapters of *AI Agent Engineering*,
built with Next.js (App Router) and deployed on Vercel.

## Local development

```bash
npm install
npm run sync   # copy chapters from ../book into content/
npm run dev    # http://localhost:3000
```

## How content works

The chapters live in `../book` (the book project's source of truth).
`npm run sync` copies the eight published chapters into `content/`, driven by the
manifest in [`lib/chapters.ts`](lib/chapters.ts). The copied files in `content/`
are committed so the site builds standalone on Vercel.

**When you edit a chapter in `../book`, re-run `npm run sync` and commit.**

To publish a different set of chapters, edit the `CHAPTERS` array in
`lib/chapters.ts` and the `FILES` list in `scripts/sync-content.mjs`, then sync.

## Comments

Each chapter has a [giscus](https://giscus.app) comment widget (backed by GitHub
Discussions) and an "open an issue" link. To enable them:

1. Push this repo to GitHub and enable **Discussions**.
2. Install the [giscus app](https://github.com/apps/giscus) on the repo.
3. Visit <https://giscus.app>, enter your repo, and copy the generated IDs.
4. Copy `.env.example` to `.env.local` and fill in the `NEXT_PUBLIC_GISCUS_*`
   and `NEXT_PUBLIC_GITHUB_REPO_URL` values. Set the same vars in the Vercel
   project settings.

Until configured, the widget shows a short placeholder and the build still works.

## Deploy to Vercel

1. Push to GitHub.
2. Import the repo at [vercel.com/new](https://vercel.com/new). The project root
   is this `website/` folder — set the **Root Directory** to `website` if the
   repo contains the whole book project.
3. Add the `NEXT_PUBLIC_*` environment variables.
4. Deploy. Framework preset and build command are auto-detected (`next build`).
