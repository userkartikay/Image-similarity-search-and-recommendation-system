const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const queryImage = document.querySelector('#query-image');
const searchButton = document.querySelector('#search-button');
const searchLabel = searchButton.querySelector('.button-label');
const clearButton = document.querySelector('#clear-button');
const fileName = document.querySelector('#file-name');
const steps = document.querySelectorAll('.steps li');
const statusEl = document.querySelector('#status');
const statusText = document.querySelector('#status-text');
const resultsSection = document.querySelector('#results-section');
const resultsGrid = document.querySelector('#results-grid');
const resultMeta = document.querySelector('#result-meta');
const errorMessage = document.querySelector('#error-message');
const SKELETON_COUNT = 5;
let selectedFile = null;
let previewUrl = null;

async function parseApiResponse(response) {
  const body = await response.text();
  let data = null;
  if (body.trim()) {
    try {
      data = JSON.parse(body);
    } catch {
      throw new Error(`Server returned ${response.status} ${response.statusText}.`);
    }
  }
  if (!response.ok) {
    throw new Error(data?.detail || `Request failed with status ${response.status}.`);
  }
  if (!data) {
    throw new Error('The server returned an empty response. It may be restarting, please try again.');
  }
  return data;
}

function setStatus(state, text) {
  statusEl.dataset.state = state;
  statusText.textContent = text;
}

fetch('/health')
  .then(parseApiResponse)
  .then(() => setStatus('ready', 'Ready to search'))
  .catch(() => setStatus('error', 'Server unavailable'));

function setStep(active) {
  steps.forEach((step) => {
    const n = Number(step.dataset.step);
    step.classList.toggle('active', n === active);
    step.classList.toggle('done', n < active);
  });
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
}

function formatSize(bytes) {
  return bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

function selectFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please choose a JPG, PNG, or WebP image.');
    return;
  }
  clearError();
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  queryImage.src = previewUrl;
  queryImage.hidden = false;
  dropZone.classList.add('has-image');
  fileName.textContent = `${file.name} · ${formatSize(file.size)}`;
  clearButton.hidden = false;
  searchButton.disabled = false;
  setStep(2);
}

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});
clearButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  selectFile(fileInput.files[0]);
  fileInput.value = '';
});
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));

async function shrinkImage(file, maxSide = 640) {
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    return await new Promise((resolve) => canvas.toBlob((blob) => resolve(blob || file), 'image/jpeg', 0.9));
  } catch {
    return file;
  }
}

function showSkeletons() {
  resultsGrid.innerHTML = Array.from({ length: SKELETON_COUNT }, () =>
    '<div class="result-card skeleton" aria-hidden="true"><div class="result-media"></div><div class="result-info"><div class="sk-line"></div><div class="sk-line short"></div></div></div>'
  ).join('');
  resultMeta.textContent = 'Searching the catalog...';
  resultsSection.hidden = false;
}

function setLoading(loading) {
  searchButton.disabled = loading;
  searchButton.classList.toggle('loading', loading);
  searchLabel.textContent = loading ? 'Analyzing your style...' : 'Find similar styles';
  dropZone.classList.toggle('scanning', loading);
}

searchButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  clearError();
  setLoading(true);
  setStep(2);
  showSkeletons();
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const started = performance.now();
  try {
    const formData = new FormData();
    formData.append('image', await shrinkImage(selectedFile), 'query.jpg');
    const response = await fetch('/api/search', { method: 'POST', body: formData });
    const data = await parseApiResponse(response);
    renderResults(data.matches, (performance.now() - started) / 1000);
    setStep(3);
    setStatus('ready', 'Ready to search');
  } catch (error) {
    resultsSection.hidden = true;
    showError(error.message);
    setStep(2);
  } finally {
    setLoading(false);
  }
});

function renderResults(matches, seconds) {
  resultsGrid.innerHTML = '';
  resultMeta.textContent = `${matches.length} matches · found in ${seconds.toFixed(1)}s`;
  matches.forEach((match, position) => {
    const percent = Math.max(0, Math.min(100, match.score * 100));
    const card = document.createElement('article');
    card.className = position === 0 ? 'result-card best' : 'result-card';
    card.style.animationDelay = `${position * 80}ms`;

    const media = document.createElement('div');
    media.className = 'result-media';
    const img = document.createElement('img');
    img.src = match.image_url;
    img.alt = `Catalog match ${position + 1}`;
    img.loading = 'lazy';
    const rank = document.createElement('span');
    rank.className = 'rank';
    rank.textContent = position === 0 ? 'Best match' : `#${position + 1}`;
    media.append(img, rank);

    const info = document.createElement('div');
    info.className = 'result-info';
    info.innerHTML = `<div class="score-row"><span class="score">${percent.toFixed(1)}%</span><span class="score-label">similar</span></div><div class="meter" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(0)}" aria-label="Similarity"><span style="width:${percent}%"></span></div>`;

    card.append(media, info);
    resultsGrid.appendChild(card);
  });
}
