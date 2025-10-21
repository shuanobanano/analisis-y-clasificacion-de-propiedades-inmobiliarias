const selectButton = document.getElementById('select-file');
const processButton = document.getElementById('process');
const fileNameLabel = document.getElementById('file-name');
const statusContainer = document.getElementById('status');
const resultsContainer = document.getElementById('results');

let selectedFile = null;

function renderStatus(message, type = 'info') {
  statusContainer.textContent = message;
  statusContainer.className = `status ${type}`;
}

function createProbabilityList(probabilities = {}) {
  const barato = ((probabilities.BARATO ?? 0) * 100).toFixed(1);
  const regular = ((probabilities.REGULAR ?? 0) * 100).toFixed(1);
  const caro = ((probabilities.CARO ?? 0) * 100).toFixed(1);

  return `
    <ul class="probability-list">
      <li><strong>BARATO:</strong> ${barato}%</li>
      <li><strong>REGULAR:</strong> ${regular}%</li>
      <li><strong>CARO:</strong> ${caro}%</li>
    </ul>
  `;
}

function renderResults(results = []) {
  if (!results.length) {
    resultsContainer.innerHTML = '<p>No se generaron predicciones.</p>';
    return;
  }

  const items = results
    .map((result, index) => {
      if (result.status === 'error') {
        return `
          <article class="card error">
            <h3>Propiedad ${index + 1}</h3>
            <p class="error-message">${result.error}</p>
          </article>
        `;
      }

      const confidence = (result.confidence * 100).toFixed(1);
      const pricePerM2 = result.price_per_m2.toLocaleString(undefined, {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
      });

      return `
        <article class="card">
          <header>
            <h3>Propiedad ${index + 1}</h3>
            <span class="badge ${result.prediction.toLowerCase()}">${result.prediction}</span>
          </header>
          <p><strong>Confianza:</strong> ${confidence}%</p>
          <p><strong>Precio por m²:</strong> ${pricePerM2}</p>
          <details>
            <summary>Probabilidades</summary>
            ${createProbabilityList(result.probabilities)}
          </details>
          <details>
            <summary>Reporte detallado</summary>
            <pre>${result.report}</pre>
          </details>
        </article>
      `;
    })
    .join('');

  resultsContainer.innerHTML = items;
}

selectButton.addEventListener('click', async () => {
  const filePath = await window.propertyAPI.selectFile();
  if (!filePath) {
    return;
  }
  selectedFile = filePath;
  fileNameLabel.textContent = filePath;
  renderStatus('Archivo listo para procesar', 'info');
});

processButton.addEventListener('click', async () => {
  if (!selectedFile) {
    renderStatus('❌ Debe seleccionar un archivo primero.', 'error');
    return;
  }

  renderStatus('Procesando archivo, por favor espere...', 'loading');
  resultsContainer.innerHTML = '';

  try {
    const response = await window.propertyAPI.runPrediction(selectedFile);
    if (!response) {
      renderStatus('❌ No se recibió respuesta del motor de predicción.', 'error');
      return;
    }

    if (response.status !== 'success') {
      renderStatus(response.error || '❌ Error desconocido durante la predicción', 'error');
      if (response.details) {
        resultsContainer.innerHTML = `<pre>${response.details}</pre>`;
      }
      return;
    }

    renderStatus('✅ Predicciones generadas correctamente.', 'success');
    renderResults(response.results);
  } catch (error) {
    renderStatus(`❌ Error inesperado: ${error.message}`, 'error');
  }
});
