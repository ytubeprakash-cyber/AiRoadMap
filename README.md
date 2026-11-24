# Modelcraft — Saurav's AI learning workspace

A responsive, dependency-free website to track a custom 24-week applied AI / LLM engineering roadmap. Includes five interactive labs, 120 tasks with key concepts per week, five portfolio project briefs, experiment logging, a learning journal, 20 interview questions with answer guides, confidence ratings, resources and JSON backups.

## Run locally

Use Node.js 20+:

```sh
npm run check
npm test
npm run build
npm run dev
```

Open http://localhost:3000. Alternatively serve `dist` with any static web server after building.

## Deploy on Vercel

1. Extract this folder and push its contents into a GitHub repository.
2. In Vercel, select **Add New → Project** and import that repository.
3. Select framework **Other**, build command `npm run build`, output directory `dist`. No environment variables or install dependencies are required.
4. Deploy. The included `vercel.json` sets build/output options.

Or run `npx vercel` from this folder, then `npx vercel --prod` after reviewing the preview. A Vercel account is required. This package has not been deployed for you.

## Data and backups

All progress is saved in browser localStorage under `modelcraft-v1`. No account, backend, API keys or cloud sync. Different devices, browsers and domains have separate progress. Export JSON regularly; import it on your production domain. Import asks before replacing existing progress. Private/incognito storage may be temporary. Clearing browser data removes progress. The app does not execute training jobs or verify project completion.

## Interactive labs

`labs.js` contains five in-browser simulations, each with "try this" prompts, how it maps to real systems, and a Python skeleton to rebuild it:

- **Gradient descent**: fit a line, watch the loss curve, make training diverge.
- **BPE tokenizer**: train merges on your own text and see the tokens and compression.
- **Sampling**: temperature, top-k and top-p applied to real-looking logits.
- **Retrieval for RAG**: TF-IDF + cosine ranking, the assembled prompt, and where keyword search fails.
- **Agent loop**: step through tool calling and watch the `messages` array grow.

Labs are deep-linkable, e.g. `#labs/agent`. Roadmap weeks link to the relevant lab.

## Curriculum

Custom proposed schedule, not an official course syllabus or a guarantee of interview readiness. Approximately 216 hours; extend for CS336 assignments. Core: Karpathy Zero to Hero → Stanford CS336 2025 → Hugging Face LLM/smol-course, plus LLM API, agent and MCP docs for weeks 9–12 ("Build with LLMs"). Systems-heavy GPU assignments and DPO are labeled stretch work. The plan targets applied AI/LLM engineering rather than research scientist roles. Course links were checked on 17 November 2026; availability may change.

Version 2 replaced weeks 9–12 (CS336 systems) with LLM application engineering (APIs, structured output, tool calling, agents, MCP) and week 16 with inference internals (memory math, KV cache, quantization). Any ticks saved for those weeks under v1 are kept in the backup under `archived.curriculumV1`.

Project milestones: character LM → tiny GPT from random weights → tool-using agent with MCP → small-model LoRA fine-tuning with held-out evaluation → mobile assistant with RAG and a separately hosted inference API. Do not upload confidential company data to public repositories or model hosts.

## Week 1 · Task 4 — ML math cheat sheet

| Concept | What to remember | Worked example |
| --- | --- | --- |
| **Mean** | Add the values and divide by how many there are. Use it to summarize per-example losses in a batch. | Token counts `[2, 4, 6]` → `(2 + 4 + 6) / 3 = 4`. |
| **Population variance** | Average the squared distances from the mean. LayerNorm uses variance to scale features. | `[2, 4, 6]` → `(4 + 0 + 4) / 3 = 8/3 ≈ 2.67`. `[4, 4, 4]` has the same mean and zero variance. |
| **Probability distribution** | Each probability is nonnegative and the total is 1. A language model assigns one probability per possible next token. | After `I like`, the model could give `tea` 60%, `coffee` 30%, and `water` 10%. Sampling can pick any of them. |
| **Softmax** | Convert raw scores (logits) into probabilities. Exponentiate each score, then divide by the sum of the exponentials. | Logits `[2, 1, 0]` → exponentials `[7.39, 2.72, 1.00]` → probabilities `[66.5%, 24.5%, 9.0%]`. |
| **Gradient** | The slope of the loss with respect to each parameter. A positive slope means a small increase in that parameter raises the loss. | For `L(w) = (w − 3)²`, the derivative is `2(w − 3)`. It is `+4` at `w = 5` and `−2` at `w = 2`. |
| **Gradient descent** | Update each weight with `weight − learning_rate × gradient`, then recompute the gradient at the new weights. | With learning rate `0.25`, the first update is `5 − 0.25 × 4 = 4`. After two steps, weights go `5 → 4 → 3.5` and losses go `4 → 1 → 0.25`. |

**Softmax in code.** Subtract the largest logit from every logit before exponentiating. The probabilities stay the same, and the exponentials cannot overflow.

**How it fits together.** The model produces logits. Softmax gives next-token probabilities. The loss penalizes low probability on the actual next token. Backpropagation computes the gradients, and the optimizer updates the weights.

<details>
<summary>Quick check: what happens at w = 2 with a learning rate of 0.25?</summary>

The gradient is `−2`, so the next weight is `2 − 0.25 × (−2) = 2.5`. The loss falls from `1` to `0.25`.

</details>

## Customize

- `curriculum.js`: weeks, tasks, concepts, projects, interview bank and resource links.
- `labs.js`: interactive labs (pure logic is tested by `verify.mjs`).
- `style.css`: colors and responsive layout.
- `app.js`: progress, logging, import/export and views.
- `build.mjs`: copies static production assets to `dist`.
- `verify.mjs`: run with `npm test`.

Font loading is optional; system fonts work without Google Fonts. Training/inference requires separate compute; Vercel serves this tracker frontend. Check model licenses and GPU provider costs before training.
