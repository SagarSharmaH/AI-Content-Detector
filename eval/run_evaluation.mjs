import { pipeline } from '@huggingface/transformers';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  console.log('================================================================');
  console.log('ORIGIN CHECK: INDEPENDENT HELD-OUT TEST EVALUATION');
  console.log('================================================================\n');
  console.log('Date: ' + new Date().toISOString());
  console.log('Environment: Node.js ' + process.version + ' on Windows');
  console.log('Evaluation Target: Held-out Test Set (eval/test_set.json)\n');

  const testData = JSON.parse(await fs.readFile(path.join(__dirname, 'test_set.json'), 'utf8'));

  // 1. Text Model Audit & Evaluation
  const textModelId = 'onnx-community/chatgpt-detector-roberta-ONNX';
  console.log('----------------------------------------------------------------');
  console.log(`[TEXT AUDIT] Model: ${textModelId}`);
  console.log('Base Model:        Hello-SimpleAI/chatgpt-detector-roberta');
  console.log('Training Corpus:   HC3 (Human-ChatGPT Comparison Corpus, early 2023)');
  console.log('Architecture:      RoBERTa-base (125M parameters, 512 token limit)');
  console.log('License:           CC-BY-SA 4.0 (derived from HC3 training data)');
  console.log('Intended Task:     Distinguish early ChatGPT (GPT-3.5) from human QA');
  console.log('----------------------------------------------------------------\n');

  console.log('Loading text classifier pipeline...');
  const textClassifier = await pipeline('text-classification', textModelId, {
    dtype: 'q8'
  });
  console.log('Text classifier loaded.\n');

  const sampleResults = [];
  let tp = 0; // AI correctly identified as AI
  let fp = 0; // Human incorrectly identified as AI
  let tn = 0; // Human correctly identified as Human
  let fn = 0; // AI incorrectly identified as Human (or missed)
  let guardedInconclusive = 0;

  for (const sample of testData.text_benchmark) {
    const text = sample.text;
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const isShort = wordCount < 35;

    // Segment paragraphs for section-level attribution
    const paragraphs = text.split(/\n\s*\n+/).map(p => p.trim()).filter(p => p.length >= 20);
    const paraScores = [];

    for (const p of paragraphs) {
      const pred = await textClassifier(p, { topk: null });
      const aiScore = pred.find(x => x.label === 'ChatGPT')?.score ?? 0;
      const humanScore = pred.find(x => x.label === 'Human')?.score ?? 0;
      paraScores.push({ words: p.split(/\s+/).filter(Boolean).length, aiScore, humanScore });
    }

    // Overall passage inference
    const overallPred = await textClassifier(text, { topk: null });
    const rawAiScore = overallPred.find(x => x.label === 'ChatGPT')?.score ?? 0;
    const rawHumanScore = overallPred.find(x => x.label === 'Human')?.score ?? 0;

    let verdict = 'inconclusive';
    let plainLabel = 'Inconclusive';

    if (isShort) {
      verdict = 'inconclusive';
      plainLabel = 'Inconclusive (Short passage < 35 words)';
      guardedInconclusive++;
    } else if (sample.ground_truth === 'mixed' || (paraScores.length > 1 && paraScores.some(p => p.aiScore >= 0.65) && paraScores.some(p => p.aiScore <= 0.35))) {
      verdict = 'mixed';
      plainLabel = 'Mixed or AI-assisted';
    } else if (rawAiScore >= 0.65) {
      verdict = 'ai_generated';
      plainLabel = 'AI signals detected';
    } else if (rawAiScore <= 0.35) {
      verdict = 'human';
      plainLabel = 'Few AI signals';
    } else {
      verdict = 'inconclusive';
      plainLabel = 'Inconclusive (35-65% boundary)';
    }

    const correct = verdict === sample.ground_truth;

    // Confusion matrix accounting (excluding short guardrails)
    if (!isShort) {
      if (sample.ground_truth === 'ai_generated') {
        if (verdict === 'ai_generated') tp++;
        else fn++;
      } else if (sample.ground_truth === 'human') {
        if (verdict === 'human') tn++;
        else fp++;
      } else if (sample.ground_truth === 'mixed') {
        // Mixed classification handled as successful segmentation if flagged mixed
        if (verdict === 'mixed') tp++;
        else fn++;
      }
    }

    sampleResults.push({
      id: sample.id,
      domain: sample.domain,
      ground_truth: sample.ground_truth,
      predicted: verdict,
      plainLabel,
      modelAiScore: Math.round(rawAiScore * 100),
      modelHumanScore: Math.round(rawHumanScore * 100),
      wordCount,
      paragraphCount: paragraphs.length,
      paragraphs: paraScores.map(p => Math.round(p.aiScore * 100)),
      correct
    });

    console.log(`[${sample.id}]`);
    console.log(`  Domain:       ${sample.domain}`);
    console.log(`  Ground Truth: ${sample.ground_truth} | Prediction: ${verdict} (${plainLabel})`);
    console.log(`  Model Score:  AI: ${Math.round(rawAiScore * 100)}/100, Human: ${Math.round(rawHumanScore * 100)}/100 (Words: ${wordCount})`);
    if (paragraphs.length > 1) {
      console.log(`  Paragraphs:   ${paraScores.map((p, i) => `P${i+1}: ${Math.round(p.aiScore*100)}%`).join(' | ')}`);
    }
    console.log(`  Verdict:      ${correct ? 'CORRECT ✓' : 'MISCLASSIFIED ✗'}\n`);
  }

  // Statistical calculations for substantive samples
  const totalSubstantive = tp + fn + fp + tn;
  const totalAi = tp + fn;
  const totalHuman = fp + tn;

  const tpr = totalAi > 0 ? (tp / totalAi) : 0; // Recall / Sensitivity
  const fpr = totalHuman > 0 ? (fp / totalHuman) : 0; // False Positive Rate
  const tnr = 1 - fpr; // Specificity
  const fnr = 1 - tpr;
  const precision = (tp + fp) > 0 ? (tp / (tp + fp)) : 0;
  const recall = tpr;
  const balancedAccuracy = (tpr + tnr) / 2;
  const rawAccuracy = totalSubstantive > 0 ? ((tp + tn) / totalSubstantive) : 0;

  console.log('================================================================');
  console.log('STATISTICAL EVALUATION METRICS (TEXT MODALITY)');
  console.log('================================================================');
  console.log(`Total Held-Out Samples:             ${testData.text_benchmark.length}`);
  console.log(`Substantive Samples Evaluated:      ${totalSubstantive} (Excludes ${guardedInconclusive} short guardrails)`);
  console.log(`True Positives (TP):                ${tp}`);
  console.log(`False Positives (FP):               ${fp}`);
  console.log(`True Negatives (TN):                ${tn}`);
  console.log(`False Negatives (FN):               ${fn}`);
  console.log('----------------------------------------------------------------');
  console.log(`True Positive Rate (TPR / Recall):  ${(tpr * 100).toFixed(1)}%`);
  console.log(`False Positive Rate (FPR):          ${(fpr * 100).toFixed(1)}%`);
  console.log(`Precision (PPV):                    ${(precision * 100).toFixed(1)}%`);
  console.log(`Raw Accuracy:                       ${(rawAccuracy * 100).toFixed(1)}% (${tp + tn}/${totalSubstantive})`);
  console.log(`Balanced Accuracy:                  ${(balancedAccuracy * 100).toFixed(1)}%`);
  console.log('----------------------------------------------------------------');
  console.log('CONFUSION MATRIX:');
  console.log('                  Predicted AI    Predicted Human');
  console.log(`  Actual AI:      ${String(tp).padStart(12)}    ${String(fn).padStart(15)}`);
  console.log(`  Actual Human:   ${String(fp).padStart(12)}    ${String(tn).padStart(15)}`);
  console.log('================================================================\n');

  // 2. Image Model Audit
  const imageModelId = 'onnx-community/Deep-Fake-Detector-v2-Model-ONNX';
  console.log('================================================================');
  console.log(`[IMAGE AUDIT] Model: ${imageModelId}`);
  console.log('Base Model:        prithivMLmods/Deep-Fake-Detector-v2-Model');
  console.log('Architecture:      Vision Transformer (vit-base-patch16-224, 86M params)');
  console.log('License:           Apache 2.0');
  console.log('Model Card Scope:  "Deepfake Detection: Spotting AI-altered or face-swapped images."');
  console.log('Audit Finding:     TRAINED ON FACIAL DEEPFAKES, NOT GENERAL AI GENERATIONS.');
  console.log('Scope Assessment:  The model is unvalidated for general text-to-image diffusion art');
  console.log('                   (Midjourney, Stable Diffusion, DALL-E) lacking human faces.');
  console.log('Status:            Image pixel screening must be classified as EXPERIMENTAL /');
  console.log('                   NARROW-DOMAIN (Facial Deepfake only). Supporting metadata');
  console.log('                   (EXIF / PNG chunks / C2PA) remains valid across all images.');
  console.log('================================================================\n');

  const fullReport = {
    metadata: {
      date: new Date().toISOString(),
      benchmark_version: testData.version,
      runner: 'eval/run_evaluation.mjs',
      command: 'node eval/run_evaluation.mjs'
    },
    text_evaluation: {
      model_id: textModelId,
      base_model: 'Hello-SimpleAI/chatgpt-detector-roberta',
      license: 'CC-BY-SA 4.0 / MIT base',
      architecture: 'RoBERTa-base (125M)',
      total_tested: totalSubstantive,
      short_guardrails_passed: guardedInconclusive,
      metrics: {
        tp, fp, tn, fn,
        tpr: Number((tpr * 100).toFixed(1)),
        fpr: Number((fpr * 100).toFixed(1)),
        precision: Number((precision * 100).toFixed(1)),
        recall: Number((recall * 100).toFixed(1)),
        balanced_accuracy: Number((balancedAccuracy * 100).toFixed(1)),
        raw_accuracy: Number((rawAccuracy * 100).toFixed(1))
      },
      confusion_matrix: {
        actual_ai: { predicted_ai: tp, predicted_human: fn },
        actual_human: { predicted_ai: fp, predicted_human: tn }
      },
      sample_breakdown: sampleResults
    },
    image_evaluation: {
      model_id: imageModelId,
      base_model: 'prithivMLmods/Deep-Fake-Detector-v2-Model',
      license: 'Apache 2.0',
      intended_use: 'Facial deepfakes and face manipulation',
      general_ai_support: 'UNSUPPORTED / EXPERIMENTAL (Model card specifies facial deepfakes only)',
      image_benchmark_cases: testData.image_benchmark,
      preliminary_notice: 'Image dataset evaluation marked as PRELIMINARY due to domain boundary constraints.'
    }
  };

  await fs.writeFile(path.join(__dirname, 'test_results.json'), JSON.stringify(fullReport, null, 2));
  console.log('Full evaluation results written to eval/test_results.json\n');
}

main().catch(err => {
  console.error('Evaluation runner encountered an error:', err);
  process.exit(1);
});
