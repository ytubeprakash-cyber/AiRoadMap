# Modelcraft — Saurav's AI learning workspace

A responsive, dependency-free website to track a custom 24-week applied AI / LLM engineering roadmap. Includes 120 tasks, four portfolio project briefs, experiment logging, a learning journal, 15 interview questions with answer guides, confidence ratings, resources and JSON backups.

## Run locally

Use Node.js 20+:

```sh
npm run check
npm run build
npm run dev
```

Open http://localhost:3000. Alternatively serve `dist` with any static web server after building.

## Deploy on Vercel

1. Extract this folder and push its contents into a GitHub repository.
2. In Vercel, select **Add New → Project** and import that repository.
3. Select framework **Other**, build command `npm run build`, output directory `dist`. No environment variables or install dependencies are required.
4. Deploy. The included `vercel.json` sets build/output options.

Or run `npx vercel` from this folder, then `npx vercel --prod` after reviewing the preview. A Vercel account is required. The source is available in this repository. Import it into Vercel to deploy.

## Data and backups

All progress is saved in browser localStorage under `modelcraft-v1`. No account, backend, API keys or cloud sync. Different devices, browsers and domains have separate progress. Export JSON regularly; import it on your production domain. Import asks before replacing existing progress. Private/incognito storage may be temporary. Clearing browser data removes progress. The app does not execute training jobs or verify project completion.

## Curriculum

Custom proposed schedule, not an official course syllabus or a guarantee of interview readiness. Approximately 216 hours; extend for CS336 assignments. Core: Karpathy Zero to Hero → Stanford CS336 2025 → Hugging Face LLM/smol-course. Systems-heavy GPU assignments and DPO are labeled stretch work. The plan targets applied AI/LLM engineering rather than research scientist roles. Course links were checked on 17 November 2026; availability may change.

Project milestones: character LM → tiny GPT from random weights → small-model LoRA fine-tuning with held-out evaluation → mobile assistant with RAG and a separately hosted inference API. Do not upload confidential company data to public repositories or model hosts.

## Customize

- `curriculum.js`: weeks, tasks, projects, interview bank and resource links.
- `style.css`: colors and responsive layout.
- `app.js`: progress, logging, import/export and views.
- `build.mjs`: copies static production assets to `dist`.

Font loading is optional; system fonts work without Google Fonts. Training/inference requires separate compute; Vercel serves this tracker frontend. Check model licenses and GPU provider costs before training.
