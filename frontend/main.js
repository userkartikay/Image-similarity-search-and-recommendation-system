const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const chooseButton = document.querySelector('#choose-button');
const searchButton = document.querySelector('#search-button');
const queryUpscale = document.querySelector('#query-upscale');
const fileName = document.querySelector('#file-name');
const queryPreview = document.querySelector('#query-preview');
const queryState = document.querySelector('#query-state');
const resultsSection = document.querySelector('#results-section');
const resultsGrid = document.querySelector('#results-grid');
const resultCount = document.querySelector('#result-count');
const errorMessage = document.querySelector('#error-message');
let selectedFile = null;
let enhancementEnabled = false;

fetch('/health')
  .then((response) => response.json())
  .then((status) => { enhancementEnabled = status.upscaler_ready; })
  .catch(() => {});

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
}

function selectFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please choose a JPG, PNG, or WebP image.');
    return;
  }
  selectedFile = file;
  fileName.textContent = `${file.name} / ${(file.size / 1024).toFixed(0)} KB`;
  searchButton.disabled = false;
  queryState.textContent = 'Ready';
  const reader = new FileReader();
  reader.onload = () => {
    queryPreview.classList.remove('empty');
    queryPreview.innerHTML = `<img src="${reader.result}" alt="Selected query image">`;
  };
  reader.readAsDataURL(file);
  clearError();
}

chooseButton.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (event) => {
  if (event.target !== chooseButton) fileInput.click();
});
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));

searchButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  const formData = new FormData();
  formData.append('image', selectedFile);
  searchButton.disabled = true;
  searchButton.innerHTML = 'Searching...';
  queryState.textContent = 'Processing';
  clearError();
  try {
    const response = await fetch(`/api/search?upscale=${queryUpscale.checked}`, { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Search failed.');
    queryPreview.innerHTML = `<img src="${data.query_preview}" alt="Processed query image">`;
    renderResults(data.matches);
    queryState.textContent = 'Complete';
  } catch (error) {
    showError(error.message);
    queryState.textContent = 'Error';
  } finally {
    searchButton.disabled = false;
    searchButton.innerHTML = 'Search catalog <span aria-hidden="true">↗</span>';
  }
});

function renderResults(matches) {
  resultsGrid.innerHTML = '';
  resultCount.textContent = `${matches.length} results / ranked by cosine similarity`;
  matches.forEach((match, position) => {
    const card = document.createElement('article');
    card.className = 'result-card';
    card.style.animationDelay = `${position * 70}ms`;
    const enhanceControl = enhancementEnabled ? '<button class="enhance-button" type="button">Enhance detail</button>' : '';
    card.innerHTML = `<img class="result-image" src="${match.image_url}" alt="Catalog match ${position + 1}" loading="lazy"><div class="result-info"><div class="result-number">MATCH / 0${position + 1}</div><div class="score"><span>${(match.score * 100).toFixed(1)}%</span><small>SIMILARITY</small></div>${enhanceControl}</div>`;
    const enhanceButton = card.querySelector('.enhance-button');
    if (!enhanceButton) {
      resultsGrid.appendChild(card);
      return;
    }
    enhanceButton.addEventListener('click', (event) => {
      event.currentTarget.textContent = 'Enhancing...';
      const resultImage = card.querySelector('.result-image');
      resultImage.src = `${match.image_url}?upscale=true`;
      resultImage.addEventListener('load', () => { event.currentTarget.textContent = 'Enhanced'; }, { once: true });
      resultImage.addEventListener('error', () => {
        event.currentTarget.textContent = 'Enhance failed';
        event.currentTarget.disabled = false;
        resultImage.src = match.image_url;
      }, { once: true });
    });
    resultsGrid.appendChild(card);
  });
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
