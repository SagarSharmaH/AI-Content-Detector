# Origin Check — AI Content Detector Demo

Origin Check is a free, privacy-first web application for demonstrating AI text detection using open-source models.

It requires **no paid APIs**, **no credit cards**, and **no per-scan vendor fees**. It runs 100% locally out of the box using a pre-cached open-source RoBERTa model (`Hello-SimpleAI/chatgpt-detector-roberta`).

---

## Quick Start (How to Start the Demo)

### 1. Prerequisites
- Node.js 20 or newer installed.
- No API keys or accounts required.

### 2. Start the Server
Open a terminal in this project folder and run:

```bash
npm start
```

Open your browser and navigate to:
**[http://localhost:4173](http://localhost:4173)**

---

## How to Test the Demo (During Your Presentation)

We have built ready-to-click sample buttons directly into the page so you can demonstrate real detector behavior in seconds:

1. **Test AI-Generated Text:**
   - Click the **`AI ESSAY ↗`** button below the text box.
   - Click **`CHECK CONTENT →`**.
   - **Result:** Displays **`AI signals detected`** with a model score of **~96 / 100**.
   - *Explanation to audience:* This sample is a classic formulaic expository passage typical of early GPT models with structured transitional markers.

2. **Test Human-Written Text:**
   - Click the **`HUMAN BLOG ↗`** button below the text box.
   - Click **`CHECK CONTENT →`**.
   - **Result:** Displays **`Few AI signals`** with a model score of **~01 / 100**.
   - *Explanation to audience:* This sample is an authentic human personal narrative featuring idiosyncratic idioms and informal pacing that statistical models classify as human.

3. **Test Hybrid (Mixed Human + AI) Text:**
   - Click the **`HYBRID ↗`** button.
   - Click **`CHECK CONTENT →`**.
   - **Result:** Displays **`Mixed or AI-assisted`** and breaks down the passage paragraph-by-paragraph to show which section triggered the signal.

4. **Test Short Text Guardrail:**
   - Type a short sentence like *"Please send me the notes from today's meeting."* (under 35 words).
   - Click **`CHECK CONTENT →`**.
   - **Result:** Displays **`Inconclusive (Passage too short)`**.
   - *Explanation to audience:* AI detectors require statistical entropy (100+ words) to measure perplexity. Short snippets produce erratic, uncalibrated scores and are safely guarded as inconclusive.

---

## Does It Need a Free API Key?

**No. It runs 100% locally with zero API keys required.**

The local server runs the open-source RoBERTa checkpoint natively on your CPU in ~50ms using pre-cached local weights.

*(Optional)* If you ever want to connect a free cloud API, **Hugging Face** provides a 100% free serverless inference API with no credit card required:
1. Sign up for free at [huggingface.co](https://huggingface.co).
2. Go to **Settings -> Access Tokens -> Create New Token** (Role: `Read`).
3. Add `HUGGINGFACE_API_KEY=hf_...` to `.env`.
4. The server will automatically use the cloud API, keeping your token safe on the server and completely hidden from the browser.

---

## Key Limitations to Explain During Your Presentation

Be completely transparent about the real capabilities and limitations of AI detection:

1. **Model Training Distribution:**
   - The detector is based on `Hello-SimpleAI/chatgpt-detector-roberta` (trained on the 2023 HC3 dataset).
   - It performs well on classic, formulaic GPT-3.5/GPT-4 expository essays.
2. **High False Negative Rate on Modern LLMs & Paraphrasing:**
   - Outputs from newer models (such as Claude 3.5, GPT-4o, and Llama 3) and lightly paraphrased or human-edited text frequently register as **`Few AI signals`** (87.5% false-negative rate on our held-out test suite).
3. **False Positives on Formal Human Writing:**
   - Historical literature, elevated rhetoric (like Lincoln's Gettysburg Address), and formal academic prose often exhibit low perplexity and can be falsely flagged as AI (16.7% false-positive rate on human tests).
4. **Never Proof of Authorship:**
   - Detector scores are statistical clues—never conclusive proof. Emphasize that automated detectors should never be used as singular evidence in academic, legal, or disciplinary proceedings.

---

## Evaluation Benchmark

To verify performance, run the held-out benchmark:

```bash
npm run eval
```

See [`EVALUATION.md`](EVALUATION.md) for full statistical tables, confusion matrices, and empirical logs.
