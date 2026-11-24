// Origin Check - Simple Demo App
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

// Configure Transformers.js for browser fallback
env.allowLocalModels = false;
env.useBrowserCache = true;

const TEXT_MODEL_ID = 'onnx-community/chatgpt-detector-roberta-ONNX';
const IMAGE_MODEL_ID = 'onnx-community/Deep-Fake-Detector-v2-Model-ONNX';

// DOM elements
const input = document.querySelector('#content-input');
const count = document.querySelector('#char-count');
const analyzeButton = document.querySelector('#analyze-button');
const fileInput = document.querySelector('#file-input');
const fileList = document.querySelector('#file-list');
const pasteTab = document.querySelector('#paste-tab');
const uploadTab = document.querySelector('#upload-tab');
const pasteView = document.querySelector('#paste-view');
const uploadView = document.querySelector('#upload-view');
const errorMessage = document.querySelector('#error-message');
const progressContainer = document.querySelector('#progress-container');
const progressText = document.querySelector('#progress-text');
const progressPct = document.querySelector('#progress-pct');
const progressFill = document.querySelector('#progress-fill');

let selectedFiles = [];
let selectedMode = 'paste';
let browserTextPipeline = null;
let browserImagePipeline = null;

// Demo samples
const SAMPLES = {
  // Proven sample that scores ~96% AI on Hello-SimpleAI/chatgpt-detector-roberta
  ai: `The rapid advancement of renewable energy technologies has transformed the global dialogue surrounding sustainable development. In recent years, developing nations have increasingly recognized the transformative potential of decentralized solar and wind infrastructure to address chronic energy poverty. By bypassing conventional centralized fossil-fuel grids, these nations can foster economic resilience while simultaneously mitigating environmental degradation. Furthermore, international financing mechanisms and technological transfers play a pivotal role in accelerating adoption rates across rural communities. However, substantial challenges remain, including high upfront capital requirements, limited domestic technical capacity, and regulatory bottlenecks. In conclusion, navigating these hurdles will require concerted collaboration between governments, private enterprises, and international stakeholders to ensure that the transition to clean energy is equitable, scalable, and enduring.`,

  // Proven sample that scores ~1% AI on Hello-SimpleAI/chatgpt-detector-roberta
  human: `Last Thursday, my old espresso machine gave up with an alarming hiss and a puddle of lukewarm brown water all over the kitchen counter. I bought that machine in college for eighty bucks at a flea market, and it had survived four moves, two roommates who didn't know what descaling was, and at least three accidental drops off the counter. Trying to replace it was an absolute nightmare. Every modern machine seems to need a Bluetooth app, a cloud account, and proprietary plastic pods. I don't need my coffee maker to ping a server in Ohio just to pull a double shot before seven in the morning. I ended up finding an all-metal manual lever machine at an estate sale down on 4th Street, and honestly, the coffee has never tasted better.`,

  // Hybrid document: Paragraph 1 is human, Paragraph 2 is AI
  mixed: `When on board H.M.S. Beagle, as naturalist, I was much struck with certain facts in the distribution of the organic beings inhabiting South America, and in the geological relations of the present to the past inhabitants of that continent. These facts seemed to throw some light on the origin of species—that mystery of mysteries, as it has been called by one of our greatest philosophers.\n\nThe rapid advancement of renewable energy technologies has transformed the global dialogue surrounding sustainable development. In recent years, developing nations have increasingly recognized the transformative potential of decentralized solar and wind infrastructure to address chronic energy poverty. By bypassing conventional centralized fossil-fuel grids, these nations can foster economic resilience while simultaneously mitigating environmental degradation. In conclusion, navigating these hurdles will require concerted collaboration.`
};

function setMode(mode) {
  selectedMode = mode;
  const paste = mode === 'paste';
  pasteTab.classList.toggle('active', paste);
  uploadTab.classList.toggle('active', !paste);
  pasteTab.setAttribute('aria-selected', String(paste));
  uploadTab.setAttribute('aria-selected', String(!paste));
  pasteView.classList.toggle('hidden', !paste);
  uploadView.classList.toggle('hidden', paste);
  updateButton();
  clearError();
}

pasteTab.addEventListener('click', () => setMode('paste'));
uploadTab.addEventListener('click', () => setMode('upload'));

function updateStats() {
  const chars = input.value.length;
  const words = input.value.trim() ? input.value.trim().split(/\s+/).length : 0;
  count.textContent = `${chars.toLocaleString()} CHARS · ${words.toLocaleString()} WORDS`;
  updateButton();
}
input.addEventListener('input', updateStats);

// Quick sample demonstration buttons
document.querySelector('#sample-ai')?.addEventListener('click', () => {
  input.value = SAMPLES.ai;
  updateStats();
  clearError();
});
document.querySelector('#sample-human')?.addEventListener('click', () => {
  input.value = SAMPLES.human;
  updateStats();
  clearError();
});
document.querySelector('#sample-mixed')?.addEventListener('click', () => {
  input.value = SAMPLES.mixed;
  updateStats();
  clearError();
});
document.querySelector('#clear-text-btn')?.addEventListener('click', () => {
  input.value = '';
  updateStats();
  clearError();
});

function updateButton() {
  const words = input.value.trim() ? input.value.trim().split(/\s+/).length : 0;
  const validInput = selectedMode === 'paste' ? words >= 5 : selectedFiles.length > 0;
  analyzeButton.disabled = !validInput;
}

function clearError() {
  errorMessage.textContent = '';
}

function showProgress(text, pct = null) {
  progressContainer.classList.remove('hidden');
  progressText.textContent = text;
  if (pct !== null) {
    const val = Math.min(100, Math.max(0, Math.round(pct)));
    progressPct.textContent = `${val}%`;
    progressFill.style.width = `${val}%`;
  } else {
    progressPct.textContent = '…';
    progressFill.style.width = '100%';
  }
}

function hideProgress() {
  progressContainer.classList.add('hidden');
  progressFill.style.width = '0%';
}

// Drag & drop handling
const dropzone = document.querySelector('#dropzone');
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener('change', () => addFiles(fileInput.files));
['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, e => {
  e.preventDefault();
  dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, e => {
  e.preventDefault();
  dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', e => addFiles(e.dataTransfer.files));

function addFiles(files) {
  clearError();
  const allowed = /\.(txt|md|csv|pdf|docx|png|jpe?g|webp|gif)$/i;
  for (const file of files) {
    if (file.size > 25 * 1024 * 1024) {
      errorMessage.textContent = `${file.name} exceeds the 25 MB limit.`;
      continue;
    }
    if (!allowed.test(file.name)) {
      errorMessage.textContent = `${file.name} is not a supported format.`;
      continue;
    }
    if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
      selectedFiles.push(file);
    }
  }
  renderFiles();
  updateButton();
}

function renderFiles() {
  fileList.innerHTML = '';
  selectedFiles.forEach((file, index) => {
    const row = document.createElement('div');
    row.className = 'file-item';
    const name = document.createElement('span');
    const sizeKB = (file.size / 1024).toFixed(0);
    name.textContent = `${file.name} · ${sizeKB} KB`;
    const remove = document.createElement('button');
    remove.className = 'text-button';
    remove.type = 'button';
    remove.textContent = 'REMOVE';
    remove.addEventListener('click', () => {
      selectedFiles.splice(index, 1);
      renderFiles();
      updateButton();
    });
    row.append(name, remove);
    fileList.append(row);
  });
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) return resolve();
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.append(script);
  });
}

async function extractDocumentText(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (['txt', 'md', 'csv'].includes(ext)) {
    return file.text();
  }
  if (ext === 'docx') {
    showProgress('Reading Word document...');
    await loadScript('https://cdnjs.cloudflare.com/ajax/libs/mammoth/1.8.0/mammoth.browser.min.js');
    const arrayBuffer = await file.arrayBuffer();
    const result = await window.mammoth.extractRawText({ arrayBuffer });
    return result.value || '';
  }
  if (ext === 'pdf') {
    showProgress('Reading PDF text layer...');
    const pdfjs = await import('https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs');
    pdfjs.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs';
    const arrayBuffer = await file.arrayBuffer();
    const doc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
    let fullText = '';
    for (let pageNo = 1; pageNo <= doc.numPages; pageNo++) {
      const page = await doc.getPage(pageNo);
      const content = await page.getTextContent();
      const pageText = content.items.map(item => item.str).join(' ');
      fullText += pageText + '\n\n';
    }

    if (fullText.trim().length < 35) {
      showProgress('Scanned PDF detected. Starting in-browser Tesseract.js OCR...');
      await loadScript('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js');
      let ocrText = '';
      for (let pageNo = 1; pageNo <= Math.min(doc.numPages, 3); pageNo++) {
        showProgress(`Running OCR on scanned page ${pageNo}...`);
        const page = await doc.getPage(pageNo);
        const viewport = page.getViewport({ scale: 1.5 });
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext('2d');
        await page.render({ canvasContext: ctx, viewport }).promise;

        const worker = await window.Tesseract.createWorker('eng');
        const ret = await worker.recognize(canvas);
        ocrText += ret.data.text + '\n\n';
        await worker.terminate();
      }
      if (ocrText.trim().length > 20) {
        return ocrText;
      }
      throw new Error('This PDF appears to be a scanned image, but OCR could not extract readable text.');
    }
    return fullText;
  }
  throw new Error(`Unsupported document format: .${ext}`);
}

async function parseImageMetadata(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const clues = [];
  let isAiByMetadata = false;

  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4E && bytes[3] === 0x47) {
    let offset = 8;
    while (offset < bytes.length - 8) {
      const length = (bytes[offset] << 24) | (bytes[offset+1] << 16) | (bytes[offset+2] << 8) | bytes[offset+3];
      const type = String.fromCharCode(bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]);
      offset += 8;
      if (type === 'tEXt' || type === 'iTXt') {
        const chunkData = new TextDecoder('iso-8859-1').decode(bytes.subarray(offset, offset + length));
        if (/parameters|prompt|workflow|Stable Diffusion|ComfyUI|NovelAI/i.test(chunkData)) {
          isAiByMetadata = true;
          const matched = chunkData.match(/(parameters|prompt|workflow|Software)[\s\S]{0,140}/i);
          clues.push(`AI Generation Parameters in PNG chunk: ${matched ? matched[0].replace(/[\r\n]+/g, ' ') : 'parameters detected'}`);
        }
      }
      offset += length + 4;
      if (type === 'IEND') break;
    }
  }

  const textDump = new TextDecoder('iso-8859-1').decode(bytes.subarray(0, Math.min(bytes.length, 65536)));
  if (/c2pa|trainedAlgorithmicMedia/i.test(textDump)) {
    isAiByMetadata = true;
    clues.push('C2PA Content Credentials detected: Declares trained algorithmic media origin.');
  }
  const exifMatch = textDump.match(/Make\x00([^\x00]{2,20})|Model\x00([^\x00]{2,30})/);
  if (exifMatch) {
    const camera = (exifMatch[1] || exifMatch[2] || '').trim();
    if (camera) clues.push(`Camera EXIF hardware signature: ${camera}`);
  }

  return { clues, isAiByMetadata };
}

// In-browser fallback pipeline (used if backend is unreachable)
async function getBrowserTextPipeline() {
  if (!browserTextPipeline) {
    browserTextPipeline = pipeline('text-classification', TEXT_MODEL_ID, {
      device: (typeof navigator !== 'undefined' && 'gpu' in navigator) ? 'webgpu' : 'wasm',
      dtype: 'q8',
      progress_callback: p => {
        if (p.status === 'progress' && p.total) {
          const pct = (p.loaded / p.total) * 100;
          showProgress(`Downloading model weights (${p.file || 'model'})...`, pct);
        }
      }
    });
  }
  return browserTextPipeline;
}

// Check content - Fast local server API with in-browser fallback
analyzeButton.addEventListener('click', async () => {
  clearError();
  analyzeButton.disabled = true;
  analyzeButton.innerHTML = 'CHECKING… <span>◌</span>';

  try {
    let result;
    if (selectedMode === 'paste') {
      const text = input.value.trim();
      const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
      if (words < 5) throw new Error('Please enter a longer text passage to check.');
      result = await analyzeTextWithFallback(text, 'Pasted Text');
    } else {
      const images = selectedFiles.filter(f => /^image\//.test(f.type) || /\.(png|jpe?g|webp|gif)$/i.test(f.name));
      const docs = selectedFiles.filter(f => !images.includes(f));

      if (images.length && docs.length) {
        throw new Error('Please check images and text documents separately.');
      }
      if (images.length > 1) {
        throw new Error('Please evaluate one image at a time.');
      }

      if (images.length === 1) {
        result = await analyzeImageFile(images[0]);
      } else {
        showProgress('Parsing document text...');
        const extractedTexts = await Promise.all(docs.map(extractDocumentText));
        const combined = extractedTexts.join('\n\n').trim();
        if (combined.length < 20) {
          throw new Error('No readable text could be extracted.');
        }
        result = await analyzeTextWithFallback(combined, docs.map(d => d.name).join(', '));
      }
    }

    renderResult(result);
  } catch (err) {
    console.error(err);
    errorMessage.textContent = err.message || 'An error occurred while evaluating the content.';
  } finally {
    hideProgress();
    analyzeButton.innerHTML = 'CHECK CONTENT <span>→</span>';
    updateButton();
  }
});

async function analyzeTextWithFallback(text, sourceName) {
  // 1. Try local server endpoint first (fastest, pre-cached, zero browser lag)
  try {
    const res = await fetch('/api/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (res.ok) {
      const data = await res.json();
      data.sourceName = sourceName;
      return data;
    }
  } catch {
    // Server endpoint not reachable (e.g. running statically without Node)
  }

  // 2. In-browser fallback via Transformers.js
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  if (wordCount < 35) {
    return {
      kind: 'text',
      provider: 'Client-side WebAssembly',
      model: 'Hello-SimpleAI/chatgpt-detector-roberta',
      sourceName,
      wordCount,
      isShort: true,
      overallScore: null,
      sections: []
    };
  }

  showProgress('Loading open-source RoBERTa text detector...');
  const pipe = await getBrowserTextPipeline();

  const rawParagraphs = text.split(/\n\s*\n+/).map(p => p.trim()).filter(p => p.length >= 25);
  const paragraphs = rawParagraphs.length > 0 ? rawParagraphs : [text];

  const sections = [];
  let weightedAi = 0;
  let totalWords = 0;

  for (let i = 0; i < paragraphs.length; i++) {
    const p = paragraphs[i];
    const pWords = p.split(/\s+/).filter(Boolean).length;
    showProgress(`Checking section ${i + 1} of ${paragraphs.length}...`);

    const preds = await pipe(p, { top_k: null });
    const aiItem = preds.find(x => x.label === 'ChatGPT');
    const humanItem = preds.find(x => x.label === 'Human');
    const aiScore = aiItem ? aiItem.score : 0;
    const humanScore = humanItem ? humanItem.score : 0;

    sections.push({
      index: i + 1,
      text: p,
      words: pWords,
      aiScore,
      humanScore
    });

    weightedAi += aiScore * pWords;
    totalWords += pWords;
  }

  return {
    kind: 'text',
    provider: 'Client-side WebAssembly',
    model: 'Hello-SimpleAI/chatgpt-detector-roberta',
    sourceName,
    wordCount,
    isShort: false,
    overallScore: totalWords > 0 ? (weightedAi / totalWords) : 0,
    sections
  };
}

async function analyzeImageFile(file) {
  showProgress('Inspecting image metadata...');
  const { clues, isAiByMetadata } = await parseImageMetadata(file);

  return {
    kind: 'image',
    sourceName: file.name,
    metadataClues: clues,
    isAiByMetadata
  };
}

// Render plain-language results matching audited evidence
function renderResult(result) {
  document.querySelector('#empty-result').classList.add('hidden');
  document.querySelector('#result-content').classList.remove('hidden');

  const status = document.querySelector('#result-status');
  const score = document.querySelector('#score-value');
  const scoreUnit = document.querySelector('.score-unit');
  const title = document.querySelector('#result-title');
  const summary = document.querySelector('#result-summary');
  const meter = document.querySelector('#meter-fill');
  const kicker = document.querySelector('#result-kicker');
  const caveat = document.querySelector('#caveat');
  const list = document.querySelector('#findings-list');
  const findingsCount = document.querySelector('#findings-count');

  list.innerHTML = '';
  let findings = [];
  let displayPercent = 0;
  let verdictTitle = '';
  let verdictSummary = '';

  if (result.kind === 'text') {
    if (result.isShort) {
      kicker.textContent = 'EXPERIMENTAL MODEL SIGNAL';
      scoreUnit.textContent = 'NO SCORE';
      score.textContent = '—';
      meter.style.width = '0%';
      meter.style.background = 'var(--muted)';
      status.textContent = 'INCONCLUSIVE';
      verdictTitle = 'Inconclusive (Passage too short)';
      verdictSummary = `This text contains only ${result.wordCount} words. AI detection models evaluate statistical distributions (perplexity/burstiness) and require 100+ words to measure patterns. Short inputs produce unstable, uncalibrated results.`;

      findings.push({
        badge: 'SCOPE',
        badgeClass: 'inconclusive',
        title: 'Input Size Constraint',
        desc: `Word count (${result.wordCount} words) is below the minimum threshold (35 words) required for statistical language modeling.`
      });
    } else {
      displayPercent = Math.round(result.overallScore * 100);
      kicker.textContent = 'EXPERIMENTAL MODEL SIGNAL';
      scoreUnit.textContent = 'CLASSIFIER SCORE / 100';
      score.textContent = String(displayPercent).padStart(2, '0');
      meter.style.width = `${displayPercent}%`;

      const hasHighAi = result.sections.some(s => s.aiScore >= 0.65);
      const hasLowAi = result.sections.some(s => s.aiScore <= 0.35);

      if (result.sections.length > 1 && hasHighAi && hasLowAi) {
        status.textContent = 'MIXED / AI-ASSISTED';
        verdictTitle = 'Mixed or AI-assisted';
        verdictSummary = 'Different sections of this passage yielded conflicting signals: certain paragraphs exhibit stylistic markers typical of AI, while others align with human writing.';
        meter.style.background = 'var(--orange)';
      } else if (result.overallScore >= 0.65) {
        status.textContent = 'AI SIGNALS DETECTED';
        verdictTitle = 'AI signals detected';
        verdictSummary = `The open-source RoBERTa model identified statistical stylistic patterns typical of AI-generated text across ${result.wordCount} words (Model signal: ${displayPercent}/100).`;
        meter.style.background = 'var(--orange)';
      } else if (result.overallScore <= 0.35) {
        status.textContent = 'FEW AI SIGNALS';
        verdictTitle = 'Few AI signals';
        verdictSummary = `The open-source model identified few statistical patterns typical of AI text (Model signal: ${displayPercent}/100). Note: Modern LLMs (GPT-4o, Claude 3.5) and paraphrased text often evade detection.`;
        meter.style.background = 'var(--mint)';
      } else {
        status.textContent = 'INCONCLUSIVE';
        verdictTitle = 'Inconclusive';
        verdictSummary = 'The model signal sits within the ambiguous boundary (35–65%). The statistical markers are too mixed to suggest an origin.';
        meter.style.background = 'var(--muted)';
      }

      findings.push({
        badge: 'DOC',
        badgeClass: 'inconclusive',
        title: `Analyzed document: ${result.sourceName}`,
        desc: `${result.wordCount} words across ${result.sections.length} evaluated section(s). Engine: ${result.provider || 'Local Open-Source RoBERTa'}.`
      });

      result.sections.forEach(sec => {
        const secScore = Math.round(sec.aiScore * 100);
        let secLabel = 'Inconclusive';
        let secClass = 'inconclusive';
        if (sec.aiScore >= 0.65) {
          secLabel = 'AI Signal';
          secClass = 'ai';
        } else if (sec.aiScore <= 0.35) {
          secLabel = 'Few AI Signals';
          secClass = 'human';
        }

        findings.push({
          badge: `SEC ${sec.index}`,
          badgeClass: secClass,
          title: `Section ${sec.index} · Signal Score: ${secScore}/100 (${secLabel})`,
          desc: `“${sec.text.slice(0, 150)}${sec.text.length > 150 ? '…' : ''}”`
        });
      });
    }

    caveat.textContent = `Demo Notice: Evaluated via Hello-SimpleAI/chatgpt-detector-roberta (CC-BY-SA 4.0). This model is an experimental statistical classifier. It can produce false positives on formal human writing and false negatives on modern LLMs (Claude 3.5, GPT-4o) or paraphrased text. This score is an automated clue, NEVER proof of authorship.`;

  } else {
    // Image results
    kicker.textContent = 'METADATA ORIGIN SCREENING';
    scoreUnit.textContent = 'METADATA CLUE';
    score.textContent = result.isAiByMetadata ? 'AI' : '—';
    meter.style.width = result.isAiByMetadata ? '90%' : '10%';
    meter.style.background = result.isAiByMetadata ? 'var(--orange)' : 'var(--mint)';

    if (result.isAiByMetadata) {
      status.textContent = 'AI SIGNALS DETECTED';
      verdictTitle = 'AI signals detected (Origin Metadata)';
      verdictSummary = 'Embedded metadata (PNG generation parameters, ComfyUI/Stable Diffusion prompt tags, or C2PA credentials) indicates that this image was generated synthetically.';
    } else {
      status.textContent = 'INCONCLUSIVE';
      verdictTitle = 'Inconclusive (No Generator Metadata)';
      verdictSummary = 'No embedded AI generation parameters or camera EXIF hardware signatures were found. Pixels alone cannot be reliably classified without specialized domain models.';
    }

    if (result.metadataClues.length > 0) {
      result.metadataClues.forEach((clue, idx) => {
        findings.push({
          badge: `CLUE ${idx + 1}`,
          badgeClass: /AI|C2PA|parameters/i.test(clue) ? 'ai' : 'human',
          title: 'Metadata Clue',
          desc: clue
        });
      });
    } else {
      findings.push({
        badge: 'META',
        badgeClass: 'inconclusive',
        title: 'Metadata Inspection',
        desc: 'Image header contains no camera EXIF or generator metadata (common in social media uploads where metadata is stripped).'
      });
    }

    caveat.textContent = `Demo Notice: Image screening inspects embedded header metadata and origin signatures. General pixel classification across diverse text-to-image models requires multi-gigabyte models and is not feasible in a free lightweight demo.`;
  }

  title.textContent = verdictTitle;
  summary.textContent = verdictSummary;
  findingsCount.textContent = String(findings.length).padStart(2, '0');

  findings.forEach(f => {
    const li = document.createElement('li');
    const head = document.createElement('div');
    head.className = 'finding-head';

    const badge = document.createElement('span');
    badge.className = `section-badge ${f.badgeClass}`;
    badge.textContent = f.badge;

    const strong = document.createElement('strong');
    strong.textContent = f.title;
    strong.style.color = 'var(--text)';
    strong.style.fontSize = '11px';

    head.append(badge, strong);

    const body = document.createElement('div');
    body.className = 'finding-body';
    body.textContent = f.desc;

    li.append(head, body);
    list.append(li);
  });

  document.querySelector('#result-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check local backend status
async function init() {
  try {
    const res = await fetch('/api/status');
    const info = await res.json();
    console.log('Backend status:', info);
  } catch {
    console.log('Running in standalone static mode.');
  }
  updateButton();
}
init();
