# Origin Check: Model Audit & Held-Out Evaluation Report

## 1. Executive Summary & Audit Overview

This document presents the independent, held-out evaluation and architectural audit of the open-source client-side models used in **Origin Check**. 

To prevent misleading claims, this evaluation was executed against a dedicated, held-out benchmark ([`eval/test_set.json`](eval/test_set.json)) strictly isolated from the tuning set ([`eval/tuning_set.json`](eval/tuning_set.json)) used during development.

### Key Audit Findings:
1. **Text Detection Model (`Hello-SimpleAI/chatgpt-detector-roberta`):**
   - **Narrow Training Distribution:** Trained on early 2023 ChatGPT question-answering data (HC3).
   - **Severe Out-of-Domain Generalization Failure:** On modern LLMs (GPT-4o, Claude 3.5, Llama 3, Mistral Large), the model failed to detect AI generation in **87.5% of cases** (False Negative Rate = 87.5%).
   - **Vulnerability to Paraphrasing:** Light human editing or synonym replacement resulted in a 100% evasion rate.
   - **False Positives on Human Text:** Misclassified casual human writing (16.7% False Positive Rate).
   - **Balanced Accuracy:** **47.9%** (below random chance on diverse contemporary writing).
2. **Image Screening Model (`prithivMLmods/Deep-Fake-Detector-v2-Model`):**
   - **Domain Mismatch:** The model card explicitly identifies the model as a **facial deepfake detector** (spotting manipulated human faces).
   - **Unsuitable for General AI Images:** The model is **not validated** for general text-to-image diffusion art (Midjourney landscapes, Stable Diffusion illustrations, DALL-E 3 objects).
   - **Status:** Pixel screening is designated as **Experimental / Narrow-Domain (Facial Deepfakes Only)**. Non-face images are treated as out-of-scope and output **Inconclusive**, relying on embedded origin metadata (PNG parameters, EXIF, C2PA) as supporting clues.

---

## 2. Test Execution Details

- **Execution Date:** 2026-11-23T17:44:12Z
- **Environment:** Node.js v24.19.0 (Windows x64)
- **Reproduction Command:**
  ```bash
  node eval/run_evaluation.mjs
  ```
- **Raw Results File:** [`eval/test_results.json`](eval/test_results.json)
- **Dataset File:** [`eval/test_set.json`](eval/test_set.json)

---

## 3. Dataset Integrity & Licensing

The held-out benchmark contains zero private user data and uses only publicly licensed or open-access material:

| Sample ID | Domain | Ground Truth | Source / Provenance | License |
| :--- | :--- | :--- | :--- | :--- |
| `test_human_twain` | Classic Literature | Human | Mark Twain (1884), *Adventures of Huckleberry Finn*, Ch. 1 | Public Domain |
| `test_human_darwin` | Scientific History | Human | Charles Darwin (1859), *Origin of Species*, Intro | Public Domain |
| `test_human_wiki_tectonics` | Reference / Encyclopedia | Human | Wikipedia (archived 2017), "Plate Tectonics" | CC-BY-SA 4.0 |
| `test_human_student_persuasive` | Student Writing (Pre-LLM) | Human | Hewlett Foundation ASAP Dataset (2012) | Public Research Dataset |
| `test_human_pmc_crispr` | Scientific Abstract | Human | PubMed Central (PMC5421890, 2017) | CC-BY 4.0 |
| `test_human_travel_blog` | Casual Travel Narrative | Human | Patagonia Trail Journal (archived 2014) | CC0 / Personal Archive |
| `test_ai_gpt4o_economics` | Economic Policy Analysis | AI | OpenAI GPT-4o (2024) | Synthetic Output |
| `test_ai_claude35_climate` | Urban Planning Analysis | AI | Anthropic Claude 3.5 Sonnet (2024) | Synthetic Output |
| `test_ai_llama3_philosophy` | Epistemology Essay | AI | Meta Llama 3 70B Instruct (2024) | Synthetic Output |
| `test_ai_mistral_tech` | Software Architecture | AI | Mistral Large (2024) | Synthetic Output |
| `test_ai_gpt35_history` | Historical Summary | AI | OpenAI ChatGPT (GPT-3.5, 2023) | Synthetic Output |
| `test_ai_paraphrased_light` | Lightly Paraphrased AI | AI | GPT-4o text with synonym and transition swaps | Synthetic with Human Edits |
| `test_ai_paraphrased_heavy` | Heavily Rewritten AI | AI | Claude 3.5 outline rewritten in casual human prose | Synthetic with Heavy Edits |
| `test_mixed_essay_50_50` | Hybrid Document | Mixed | 50% PMC CRISPR paper / 50% GPT-4o continuation | Composite (CC-BY + Synthetic) |
| `test_short_human_calendar` | Short Human Note (14 words) | Inconclusive | Casual meeting rescheduling message | Human Text |
| `test_short_ai_headline` | Short AI Snippet (13 words) | Inconclusive | Enterprise marketing slogan from GPT-4 | Synthetic Output |

---

## 4. Text Modality Results

### 4.1 Sample-by-Sample Evaluation

| Sample ID | Domain | Ground Truth | Model Score (AI/100) | Predicted Output | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `test_human_twain` | Classic Literature | Human | 0 / 100 | Few AI signals | **Correct** ✓ |
| `test_human_darwin` | Scientific History | Human | 0 / 100 | Few AI signals | **Correct** ✓ |
| `test_human_wiki_tectonics` | Encyclopedia | Human | 0 / 100 | Few AI signals | **Correct** ✓ |
| `test_human_student_persuasive` | Student Writing | Human | 0 / 100 | Few AI signals | **Correct** ✓ |
| `test_human_pmc_crispr` | Scientific Abstract | Human | 0 / 100 | Few AI signals | **Correct** ✓ |
| `test_human_travel_blog` | Casual Travelogue | Human | **97 / 100** | AI signals detected | **False Positive** ✗ |
| `test_ai_gpt4o_economics` | GPT-4o | AI | 0 / 100 | Few AI signals | **False Negative** ✗ |
| `test_ai_claude35_climate` | Claude 3.5 Sonnet | AI | 0 / 100 | Few AI signals | **False Negative** ✗ |
| `test_ai_llama3_philosophy` | Llama 3 70B | AI | 0 / 100 | Few AI signals | **False Negative** ✗ |
| `test_ai_mistral_tech` | Mistral Large | AI | 0 / 100 | Few AI signals | **False Negative** ✗ |
| `test_ai_gpt35_history` | ChatGPT (GPT-3.5) | AI | 0 / 100 | Few AI signals | **False Negative** ✗ |
| `test_ai_paraphrased_light` | Lightly Paraphrased | AI | 0 / 100 | Few AI signals | **False Negative** ✗ |
| `test_ai_paraphrased_heavy` | Heavily Rewritten | AI | 0 / 100 | Few AI signals | **False Negative** ✗ |
| `test_mixed_essay_50_50` | Hybrid 50/50 | Mixed | 0 / 100 | Mixed or AI-assisted | **Correct** ✓ |
| `test_short_human_calendar` (14 w) | Short Human | Inconclusive | 0 / 100 | Inconclusive | **Correct (Guardrail)** ✓ |
| `test_short_ai_headline` (13 w) | Short AI | Inconclusive | 0 / 100 | Inconclusive | **Correct (Guardrail)** ✓ |

### 4.2 Statistical Metrics

Calculated across the 14 substantive test samples (excluding the 2 short-text guardrail cases):

- **Total Substantive Samples Evaluated:** 14
- **True Positives (TP):** 1
- **False Positives (FP):** 1
- **True Negatives (TN):** 5
- **False Negatives (FN):** 7
- **True Positive Rate (TPR / Recall):** **12.5%** ($1 / 8$)
- **False Negative Rate (FNR):** **87.5%** ($7 / 8$)
- **False Positive Rate (FPR):** **16.7%** ($1 / 6$)
- **True Negative Rate (TNR / Specificity):** **83.3%** ($5 / 6$)
- **Precision (Positive Predictive Value):** **50.0%** ($1 / 2$)
- **Raw Accuracy:** **42.9%** ($6 / 14$)
- **Balanced Accuracy:** **47.9%** ($\frac{0.125 + 0.833}{2}$)

### 4.3 Confusion Matrix

```
                  Predicted AI    Predicted Human
  Actual AI:                 1                  7
  Actual Human:              1                  5
```

---

## 5. Image Modality Audit & Scope Assessment

- **Model Identifier:** `onnx-community/Deep-Fake-Detector-v2-Model-ONNX`
- **Base Architecture:** Vision Transformer (`vit-base-patch16-224`, 86M parameters)
- **License:** Apache 2.0
- **Audited Intended Use:** The official model card defines the model's intended domain as:
  > *"Deepfake Detection: Spotting AI-altered or face-swapped images. Media Authentication & Forensics: Verifying digital integrity for content moderation."*

### Audit Conclusion:
1. **Face Deepfake Specialization:** The model was trained specifically on facial manipulation datasets (such as FaceForensics++ and DFDC).
2. **Unvalidated on General Diffusion Art:** The model has **not been trained or evaluated** on text-to-image diffusion models for general photography (landscapes, architectural shots, wildlife) or non-photographic artwork (illustrations, 3D renders).
3. **Product Policy:** In accordance with our audit, image pixel classification is explicitly designated as **Experimental / Narrow-Domain**. Non-face images are treated as out-of-scope and output **Inconclusive (Narrow-Domain Model)**. General image screening relies on embedded origin metadata (PNG parameters, EXIF camera signatures, and C2PA credentials).

---

## 6. Documented Limitations & Ethical Guidelines

1. **Failure on Modern LLMs:** Early ChatGPT detectors fail to identify outputs from modern foundation models (Claude 3.5, GPT-4o, Llama 3), producing high false-negative rates.
2. **Complete Paraphrasing Vulnerability:** Swapping words or restructuring sentences completely bypasses token perplexity classifiers.
3. **False Accusations on Human Writing:** Detectors frequently flag expressive, rhythmic, or non-native human writing as AI.
4. **Not Proof of Authorship:** Origin Check outputs are automated, probabilistic screening signals. **They must never be used as singular evidence in academic integrity, disciplinary, or legal proceedings.**
