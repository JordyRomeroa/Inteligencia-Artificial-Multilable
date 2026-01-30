// ============================================================================
// CLASIFICADOR MULTILABEL - SCRIPT PRINCIPAL
// ============================================================================

// Global variables (Declared once at startup)
let currentFilename = null;
let currentPredictions = null;
let batchFiles = [];

// Classes from Flask (passed via window.classes)
// Usage: window.classes is available globally
console.log('Script.js cargado correctamente');
console.log('Clases disponibles:', typeof window.classes !== 'undefined' ? window.classes.length : 'no disponibles');

// ============================================================================
// TAB MANAGEMENT
// ============================================================================

function showTab(tabName) {
    try {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show selected tab
        const tabElement = document.getElementById(`${tabName}-tab`);
        if (tabElement) {
            tabElement.classList.add('active');
        }
        event.target.classList.add('active');
        
        // Load corrections if switching to corrections tab
        if (tabName === 'corrections') {
            loadCorrections();
        }
    } catch (error) {
        console.error('Error en showTab:', error);
        showStatus('Error al cambiar pestaña', 'error');
    }
}

// Threshold Management
function updateThreshold(value) {
    document.getElementById('thresholdValue').textContent = parseFloat(value).toFixed(2);
}

// Image Preview
function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    currentFilename = file.name;
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    
    reader.readAsDataURL(file);
    
    // Hide previous results
    document.getElementById('results').classList.add('hidden');
}

// Predict Single Image
async function predictImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showStatus('Por favor selecciona una imagen', 'error');
        return;
    }
    
    const threshold = parseFloat(document.getElementById('threshold').value);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold);
    
    showStatus('Prediciendo...', 'info');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentPredictions = data.predictions;
            displayPredictions(data.predictions);
            displayCorrectionLabels(data.predictions);
            document.getElementById('results').classList.remove('hidden');
            showStatus('Predicción completada!', 'success');
        } else {
            showStatus(data.error, 'error');
        }
    } catch (error) {
        showStatus('Error al predecir: ' + error.message, 'error');
    }
}

// Display Predictions
function displayPredictions(predictions) {
    const container = document.getElementById('predictions');
    container.innerHTML = '';
    
    if (predictions.length === 0) {
        container.innerHTML = '<p class="info">No se detectaron objetos con el threshold actual</p>';
        return;
    }
    
    predictions.forEach(pred => {
        const confidence = pred.confidence;
        let confidenceClass = 'low-confidence';
        
        if (confidence >= 0.7) {
            confidenceClass = 'high-confidence';
        } else if (confidence >= 0.4) {
            confidenceClass = 'medium-confidence';
        }
        
        const item = document.createElement('div');
        item.className = `prediction-item ${confidenceClass}`;
        item.innerHTML = `
            <div class="label">${pred.label}</div>
            <div class="confidence">${(confidence * 100).toFixed(1)}% confianza</div>
            <div class="confidence-bar">
                <div class="confidence-bar-fill" style="width: ${confidence * 100}%"></div>
            </div>
        `;
        container.appendChild(item);
    });
}

// Display Correction Labels
function displayCorrectionLabels(predictions) {
    const container = document.getElementById('correctionLabels');
    container.innerHTML = '';
    
    const predictedLabels = predictions.map(p => p.label);
    
    if (!window.classes) {
        console.error('Classes no disponibles');
        return;
    }
    
    window.classes.forEach(label => {
        const isPredicted = predictedLabels.includes(label);
        const checkbox = document.createElement('div');
        checkbox.className = `label-checkbox ${isPredicted ? 'checked' : ''}`;
        checkbox.innerHTML = `
            <input type="checkbox" id="cb-${label}" value="${label}" ${isPredicted ? 'checked' : ''}>
            <label for="cb-${label}">${label}</label>
        `;
        
        checkbox.querySelector('input').addEventListener('change', function(e) {
            if (e.target.checked) {
                checkbox.classList.add('checked');
            } else {
                checkbox.classList.remove('checked');
            }
        });
        
        container.appendChild(checkbox);
    });
}

// Save Correction
async function saveCorrection() {
    if (!currentFilename) {
        showStatus('No hay imagen cargada', 'error');
        return;
    }
    
    const checkboxes = document.querySelectorAll('#correctionLabels input[type="checkbox"]:checked');
    const correctedLabels = Array.from(checkboxes).map(cb => cb.value);
    
    if (correctedLabels.length === 0) {
        showStatus('Selecciona al menos una etiqueta correcta', 'error');
        return;
    }
    
    const data = {
        filename: currentFilename,
        corrected_labels: correctedLabels
    };
    
    showStatus('Guardando corrección...', 'info');
    
    try {
        const response = await fetch('/save_correction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showStatus('Corrección guardada! ✓', 'success');
        } else {
            showStatus(result.error, 'error');
        }
    } catch (error) {
        showStatus('Error al guardar: ' + error.message, 'error');
    }
}

// Retrain Model
async function retrainModel() {
    if (!confirm('¿Deseas reentrenar el modelo con las correcciones guardadas? Esto puede tomar varios minutos.')) {
        return;
    }
    
    showStatus('Reentrenando modelo... Por favor espera', 'warning');
    
    try {
        const response = await fetch('/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ epochs: 5 })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showStatus(`Reentrenamiento completado! Loss final: ${result.final_loss.toFixed(4)}`, 'success');
            
            // Re-predict current image
            if (currentFilename) {
                setTimeout(() => {
                    predictImage();
                }, 1000);
            }
        } else {
            showStatus(result.error, 'error');
        }
    } catch (error) {
        showStatus('Error al reentrenar: ' + error.message, 'error');
    }
}

// Batch Preview
function previewBatch(event) {
    const files = Array.from(event.target.files);
    batchFiles = files;
    
    const preview = document.getElementById('batchPreview');
    preview.innerHTML = '';
    
    files.forEach(file => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.alt = file.name;
            img.title = file.name;
            preview.appendChild(img);
        };
        reader.readAsDataURL(file);
    });
}

// Predict Batch
async function predictBatch() {
    if (batchFiles.length === 0) {
        showStatus('Por favor selecciona imágenes', 'error');
        return;
    }
    
    const threshold = parseFloat(document.getElementById('threshold').value);
    const formData = new FormData();
    
    batchFiles.forEach(file => {
        formData.append('files', file);
    });
    formData.append('threshold', threshold);
    
    showStatus(`Prediciendo ${batchFiles.length} imágenes...`, 'info');
    
    try {
        const response = await fetch('/batch_predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(data.results);
            showStatus('Predicción batch completada!', 'success');
        } else {
            showStatus(data.error, 'error');
        }
    } catch (error) {
        showStatus('Error en predicción batch: ' + error.message, 'error');
    }
}

// Display Batch Results
function displayBatchResults(results) {
    const container = document.getElementById('batchResults');
    container.innerHTML = '<h2>Resultados</h2>';
    
    results.forEach((result, idx) => {
        const item = document.createElement('div');
        item.className = 'batch-item';
        
        const predictionsHTML = result.predictions.length > 0
            ? result.predictions.map(p => 
                `<span class="label-tag">${p.label} (${(p.confidence * 100).toFixed(1)}%)</span>`
              ).join(' ')
            : '<span class="info">Sin detecciones</span>';
        
        item.innerHTML = `
            <h3>Imagen ${idx + 1}: ${result.filename}</h3>
            <div class="labels">${predictionsHTML}</div>
            <button onclick="selectBatchForCorrection(${idx}, '${result.filename}')" class="btn-secondary">
                Corregir
            </button>
        `;
        
        container.appendChild(item);
    });
}

// Select Batch Image for Correction
function selectBatchForCorrection(idx, filename) {
    // Switch to single tab
    document.querySelector('.tab-btn').click();
    
    // Set the file
    const file = batchFiles[idx];
    currentFilename = filename;
    
    // Create a DataTransfer to set the file
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    document.getElementById('imageInput').files = dataTransfer.files;
    
    // Preview and predict
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('imagePreview').innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);
    
    predictImage();
}

// Load Corrections
async function loadCorrections() {
    showStatus('Cargando correcciones...', 'info');
    
    try {
        const response = await fetch('/get_corrections');
        const data = await response.json();
        
        if (data.success) {
            displayCorrections(data.corrections);
            showStatus(`${data.total} correcciones cargadas`, 'success');
        } else {
            showStatus(data.error, 'error');
        }
    } catch (error) {
        showStatus('Error al cargar correcciones: ' + error.message, 'error');
    }
}

// Display Corrections
function displayCorrections(corrections) {
    const container = document.getElementById('correctionsList');
    
    if (corrections.length === 0) {
        container.innerHTML = '<p class="info">No hay correcciones guardadas todavía</p>';
        return;
    }
    
    container.innerHTML = '';
    
    corrections.forEach(corr => {
        const card = document.createElement('div');
        card.className = 'correction-card';
        
        const labelsHTML = corr.corrected_labels.map(label => 
            `<span class="label-tag">${label}</span>`
        ).join(' ');
        
        card.innerHTML = `
            <h4>${corr.filename}</h4>
            <div class="timestamp">${corr.timestamp || 'Sin fecha'}</div>
            <div class="labels">${labelsHTML}</div>
        `;
        
        container.appendChild(card);
    });
}

// Show Status Message
function showStatus(message, type = 'info') {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status ${type}`;
    status.classList.remove('hidden');
    
    setTimeout(() => {
        status.classList.add('hidden');
    }, 4000);
}
