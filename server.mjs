import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));

// Load .env if present
try {
  const envText = await readFile(path.join(root, '.env'), 'utf8');
  for (const line of envText.split(/\r?\n/)) {
    const entry = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (entry && !entry[1].startsWith('#') && process.env[entry[1]] == null) {
      process.env[entry[1]] = entry[2].replace(/^(['"])(.*)\1$/, '$2');
    }
  }
} catch {}

const port = Number(process.env.PORT || 4173);
const hfToken = process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN || null;

// MIME types
const mime = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.pdf': 'application/pdf',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
};

const json = (res, status, data) => {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'Access-Control-Allow-Origin': '*'
  });
  res.end(JSON.stringify(data, null, 2));
};

async function readJson(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 10 * 1024 * 1024) throw Object.assign(new Error('Payload too large'), { status: 413 });
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw Object.assign(new Error('Invalid JSON request body'), { status: 400 });
  }
}

// Local text classifier singleton
let localClassifier = null;
async function getLocalClassifier() {
  if (!localClassifier) {
    const { pipeline, env } = await import('@huggingface/transformers');
    env.localModelPath = env.cacheDir;
    try {
      localClassifier = await pipeline('text-classification', 'onnx-community/chatgpt-detector-roberta-ONNX', {
        local_files_only: true,
        dtype: 'q8'
      });
    } catch {
      localClassifier = await pipeline('text-classification', 'onnx-community/chatgpt-detector-roberta-ONNX', {
        dtype: 'q8'
      });
    }
  }
  return localClassifier;
}

// Optional free Hugging Face Serverless Inference API call (if token is configured in .env)
async function callHuggingFaceInference(passage) {
  const res = await fetch('https://api-inference.huggingface.co/models/Hello-SimpleAI/chatgpt-detector-roberta', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${hfToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ inputs: passage })
  });

  if (!res.ok) {
    throw new Error(`Hugging Face API returned status ${res.status}`);
  }

  const raw = await res.json();
  const list = Array.isArray(raw[0]) ? raw[0] : raw;
  const chatgptItem = list.find(item => item.label === 'ChatGPT' || item.label === 'LABEL_1');
  const humanItem = list.find(item => item.label === 'Human' || item.label === 'LABEL_0');

  return {
    aiScore: chatgptItem ? chatgptItem.score : 0,
    humanScore: humanItem ? humanItem.score : 0
  };
}

// Score a passage using either free HF API or local cached model
async function scorePassage(passage) {
  if (hfToken) {
    try {
      return await callHuggingFaceInference(passage);
    } catch (e) {
      console.warn('Hugging Face API request failed, falling back to local model:', e.message);
    }
  }

  const classifier = await getLocalClassifier();
  const preds = await classifier(passage, { top_k: null });
  const chatgptItem = preds.find(p => p.label === 'ChatGPT');
  const humanItem = preds.find(p => p.label === 'Human');

  return {
    aiScore: chatgptItem ? chatgptItem.score : 0,
    humanScore: humanItem ? humanItem.score : 0
  };
}

async function handleApi(req, res, pathname) {
  if (req.method === 'GET' && pathname === '/api/status') {
    return json(res, 200, {
      status: 'ready',
      mode: hfToken ? 'huggingface-free-api' : 'local-model',
      hasApiKey: Boolean(hfToken),
      model: 'Hello-SimpleAI/chatgpt-detector-roberta',
      license: 'CC-BY-SA 4.0 / MIT base',
      message: hfToken
        ? 'Using free Hugging Face Serverless API (API key securely kept on server)'
        : 'Running local open-source RoBERTa model with zero API keys required'
    });
  }

  if (req.method === 'POST' && pathname === '/api/analyze-text') {
    const body = await readJson(req);
    const text = typeof body.text === 'string' ? body.text.trim() : '';
    const words = text ? text.split(/\s+/).filter(Boolean).length : 0;

    if (words < 5) {
      return json(res, 400, { error: 'Please enter a longer text passage to check.' });
    }

    // Short-text guardrail
    if (words < 35) {
      return json(res, 200, {
        kind: 'text',
        provider: hfToken ? 'Hugging Face Free API' : 'Local RoBERTa Detector',
        model: 'Hello-SimpleAI/chatgpt-detector-roberta',
        wordCount: words,
        isShort: true,
        overallScore: null,
        sections: []
      });
    }

    // Segment into paragraphs for section breakdown
    const rawParagraphs = text.split(/\n\s*\n+/).map(p => p.trim()).filter(p => p.length >= 25);
    const paragraphs = rawParagraphs.length > 0 ? rawParagraphs : [text];

    const sections = [];
    let weightedAi = 0;
    let totalWords = 0;

    for (let i = 0; i < paragraphs.length; i++) {
      const p = paragraphs[i];
      const pWords = p.split(/\s+/).filter(Boolean).length;
      const scores = await scorePassage(p);

      sections.push({
        index: i + 1,
        text: p,
        words: pWords,
        aiScore: scores.aiScore,
        humanScore: scores.humanScore
      });

      weightedAi += scores.aiScore * pWords;
      totalWords += pWords;
    }

    const overallAiScore = totalWords > 0 ? (weightedAi / totalWords) : 0;

    return json(res, 200, {
      kind: 'text',
      provider: hfToken ? 'Hugging Face Free API' : 'Local RoBERTa Detector',
      model: 'Hello-SimpleAI/chatgpt-detector-roberta',
      wordCount: words,
      isShort: false,
      overallScore: overallAiScore,
      sections
    });
  }

  return json(res, 404, { error: 'Not found' });
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);

    if (url.pathname.startsWith('/api/')) {
      return await handleApi(req, res, url.pathname);
    }

    const relative = url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname.slice(1));
    const target = path.resolve(root, relative);

    // Prevent path traversal
    if (!target.startsWith(root + path.sep) && target !== path.join(root, 'index.html')) {
      return json(res, 403, { error: 'Forbidden' });
    }

    const info = await stat(target);
    if (!info.isFile()) {
      return json(res, 404, { error: 'Not found' });
    }

    const ext = path.extname(target).toLowerCase();
    const contentType = mime[ext] || 'application/octet-stream';

    res.writeHead(200, {
      'Content-Type': contentType,
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
      'Cache-Control': ext === '.html' ? 'no-cache' : 'public, max-age=3600'
    });

    res.end(await readFile(target));
  } catch (error) {
    if (res.headersSent) return res.end();
    json(res, error.status || (error.code === 'ENOENT' ? 404 : 500), {
      error: error.message || 'Server error'
    });
  }
});

server.listen(port, () => {
  console.log(`Origin Check running at http://localhost:${port}`);
  if (hfToken) {
    console.log('Mode: Using Hugging Face Serverless Free API (Token configured in .env)');
  } else {
    console.log('Mode: Running 100% locally with open-source RoBERTa model (Zero API keys needed)');
  }
});
