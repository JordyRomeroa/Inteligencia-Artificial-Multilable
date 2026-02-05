/**
 * Advanced YOLO Detection Script
 * - Model selection and versioning
 * - Visual prediction display
 * - Interactive correction and retraining
 */

// ===== STATE VARIABLES =====
let currentModel = null;
let currentImage = null;
let currentPredictions = null;
let selectedBbox = null;
let selectedTag = null;
let correctionCanvas = null;
let correctionCtx = null;
let isDrawing = false;
let startX, startY;

const CLASSES = ['person', 'car', 'dog'];
const COLORS = {
    'person': '#FF6B6B',  // Red
    'car': '#4ECDC4',     // Teal
    'dog': '#FFE66D'      // Yellow
};

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    loadAvailableModels();
    setupCorrectionCanvas();
    loadCorrectionStats();
});

// ===== MODEL MANAGEMENT =====
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models/list');
        const data = await response.json();
        
        const select = document.getElementById('modelSelect');
        select.innerHTML = '';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = `v${model.version} - ${model.type}`;
                if (model.is_current) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
            
            currentModel = data.models.find(m => m.is_current);
            updateModelInfo();
            updateModelList(data.models);
        }
    } catch (error) {
        showStatus('Error cargando modelos: ' + error.message, 'error');
    }
}

function updateModelList(models) {
    const list = document.getElementById('modelList');
    list.innerHTML = '';
    
    models.forEach(model => {
        const item = document.createElement('div');
        item.className = `model-item ${model.is_current ? 'active' : ''}`;
        item.onclick = () => changeModel(model.path);
        item.innerHTML = `
            <strong>v${model.version}</strong>
            <div style="font-size: 0.75rem; opacity: 0.8; margin-top: 5px;">
                ${model.type}<br>
                Precisión: ${(model.metrics?.accuracy * 100 || 0).toFixed(1)}%
            </div>
        `;
        list.appendChild(item);
    });
}

async function changeModel(modelPath) {
    try {
        showStatus('Cargando modelo...', 'info');
        const response = await fetch('/api/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath })
        });
        
        const data = await response.json();
        if (data.success) {
            currentModel = data.model;
            updateModelInfo();
            showStatus('Modelo cargado correctamente', 'success');
        } else {
            showStatus('Error cargando modelo: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

function updateModelInfo() {
    if (currentModel) {
        document.getElementById('modelVersion').textContent = `v${currentModel.version}`;
        document.getElementById('modelStatus').innerHTML = `
            <span class="status-badge ready">✓ Listo</span>
        `;
        document.getElementById('gpuStatus').textContent = currentModel.device === 'cuda' ? '✓ GPU' : '⚠ CPU';
    }
}

// ===== PREDICTION TAB =====
function onImageSelected() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = {
            file: file,
            data: e.target.result
        };
        predictImage();
    };
    reader.readAsDataURL(file);
}

async function predictImage() {
    if (!currentImage || !currentModel) {
        showStatus('Por favor selecciona una imagen y un modelo', 'error');
        return;
    }
    
    showStatus('Realizando predicción...', 'info');
    
    try {
        const formData = new FormData();
        formData.append('image', currentImage.file);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.success) {
            currentPredictions = data.detections;
            displayPredictions(data);
            showStatus('Predicción completada', 'success');
        } else {
            showStatus('Error en predicción: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

async function displayPredictions(data) {
    // Mostrar imagen con bounding boxes
    const img = new Image();
    img.onload = () => {
        const canvas = document.getElementById('predictionCanvas');
        canvas.width = img.width;
        canvas.height = img.height;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        
        // Dibujar bounding boxes
        if (data.detections && data.detections.length > 0) {
            data.detections.forEach(det => {
                const bbox = det.bbox;
                const color = COLORS[det.class] || '#999';
                
                // Dibujar rectángulo
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
                
                // Dibujar etiqueta
                const label = `${det.class.toUpperCase()} ${(det.confidence * 100).toFixed(1)}%`;
                ctx.fillStyle = color;
                ctx.fillRect(bbox.x1, bbox.y1 - 25, label.length * 8, 25);
                
                ctx.fillStyle = 'white';
                ctx.font = 'bold 14px Arial';
                ctx.fillText(label, bbox.x1 + 5, bbox.y1 - 7);
            });
        }
    };
    img.src = currentImage.data;
    
    // Mostrar lista de detecciones
    const detectionsList = document.getElementById('detectionsList');
    detectionsList.innerHTML = '';
    
    if (data.detections && data.detections.length > 0) {
        data.detections.forEach((det, idx) => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            item.innerHTML = `
                <div>
                    <div class="detection-label" style="color: ${COLORS[det.class]}">
                        ${det.class.toUpperCase()}
                    </div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 5px;">
                        Confianza: ${(det.confidence * 100).toFixed(1)}%
                    </div>
                </div>
                <button class="correction-btn" onclick="correctDetection(${idx}, '${det.class}')">
                    Corregir
                </button>
            `;
            detectionsList.appendChild(item);
        });
    } else {
        detectionsList.innerHTML = '<p style="text-align: center; color: #999;">No se detectaron objetos</p>';
    }
    
    document.getElementById('predictionResult').style.display = 'block';
}

function correctDetection(detectionIdx, currentClass) {
    switchTab('correct');
    // Cargar la imagen actual en el canvas de corrección
    setTimeout(() => {
        loadCorrectionImage();
    }, 100);
}

// ===== CORRECTION TAB =====
function setupCorrectionCanvas() {
    correctionCanvas = document.getElementById('correctionCanvas');
    correctionCtx = correctionCanvas.getContext('2d');
    
    // Mouse events para selección de bbox
    correctionCanvas.addEventListener('mousedown', startBboxSelection);
    correctionCanvas.addEventListener('mousemove', drawBboxSelection);
    correctionCanvas.addEventListener('mouseup', endBboxSelection);
    correctionCanvas.addEventListener('mouseout', endBboxSelection);
}

function onCorrectionImageSelected() {
    const fileInput = document.getElementById('correctionImageInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = {
            file: file,
            data: e.target.result
        };
        loadCorrectionImage();
    };
    reader.readAsDataURL(file);
}

function loadCorrectionImage() {
    if (!currentImage) return;
    
    const img = new Image();
    img.onload = () => {
        correctionCanvas.width = img.width;
        correctionCanvas.height = img.height;
        correctionCtx.drawImage(img, 0, 0);
        
        document.getElementById('canvasContainer').style.display = 'block';
        document.getElementById('selectionInfo').style.display = 'block';
        document.getElementById('tagSelectorGroup').style.display = 'block';
        document.getElementById('actionButtons').style.display = 'flex';
    };
    img.src = currentImage.data;
}

function startBboxSelection(e) {
    const rect = correctionCanvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    isDrawing = true;
}

function drawBboxSelection(e) {
    if (!isDrawing) return;
    
    const rect = correctionCanvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    // Redraw image
    const img = new Image();
    img.onload = () => {
        correctionCtx.drawImage(img, 0, 0);
        
        // Draw selection rectangle
        correctionCtx.strokeStyle = '#667eea';
        correctionCtx.lineWidth = 2;
        correctionCtx.setLineDash([5, 5]);
        correctionCtx.strokeRect(
            startX,
            startY,
            currentX - startX,
            currentY - startY
        );
        correctionCtx.setLineDash([]);
        
        // Update coordinates display
        document.getElementById('selectionCoords').textContent = 
            `x: ${Math.min(startX, currentX)}, y: ${Math.min(startY, currentY)}, ` +
            `w: ${Math.abs(currentX - startX)}, h: ${Math.abs(currentY - startY)}`;
    };
    img.src = currentImage.data;
}

function endBboxSelection(e) {
    if (!isDrawing) return;
    isDrawing = false;
    
    const rect = correctionCanvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    // Guardar bounding box (normalizar)
    const img = new Image();
    img.onload = () => {
        selectedBbox = {
            x1: Math.min(startX, endX) / img.width,
            y1: Math.min(startY, endY) / img.height,
            x2: Math.max(startX, endX) / img.width,
            y2: Math.max(startY, endY) / img.height
        };
        
        updateSubmitButton();
    };
    img.src = currentImage.data;
}

function selectTag(tag) {
    selectedTag = tag;
    
    // Update UI
    document.querySelectorAll('.tag-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    document.querySelector(`[data-tag="${tag}"]`).classList.add('selected');
    
    updateSubmitButton();
}

function updateSubmitButton() {
    const submitBtn = document.getElementById('submitBtn');
    if (selectedBbox && selectedTag) {
        submitBtn.disabled = false;
    } else {
        submitBtn.disabled = true;
    }
}

async function submitCorrection() {
    if (!selectedBbox || !selectedTag || !currentImage) {
        showStatus('Por favor completa la selección', 'error');
        return;
    }
    
    showStatus('Guardando corrección...', 'info');
    
    try {
        const formData = new FormData();
        formData.append('image', currentImage.file);
        formData.append('bbox', JSON.stringify(selectedBbox));
        formData.append('tag', selectedTag);
        
        const response = await fetch('/api/corrections/add', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.success) {
            showStatus('✓ Corrección guardada', 'success');
            resetCorrection();
            loadCorrectionStats();
        } else {
            showStatus('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

function resetCorrection() {
    selectedBbox = null;
    selectedTag = null;
    document.querySelectorAll('.tag-btn').forEach(btn => btn.classList.remove('selected'));
    document.getElementById('correctionImageInput').value = '';
    document.getElementById('canvasContainer').style.display = 'none';
    updateSubmitButton();
}

async function loadCorrectionStats() {
    try {
        const response = await fetch('/api/corrections/stats');
        const data = response.json();
        
        document.getElementById('correctionStatus').style.display = 'block';
        document.getElementById('totalCorrections').textContent = data.total || 0;
        
        const readyForRetrain = (data.total || 0) >= 10;
        document.getElementById('readyForRetrain').textContent = readyForRetrain ? 'SÍ ✓' : 'No (mín. 10)';
        
        if (readyForRetrain) {
            document.getElementById('retrainBtn').style.display = 'block';
        }
    } catch (error) {
        console.error('Error cargando estadísticas:', error);
    }
}

async function submitRetrain() {
    const confirm = window.confirm('¿Iniciar reentrenamiento con las correcciones guardadas? Esto puede tomar varios minutos.');
    if (!confirm) return;
    
    document.getElementById('retrainStatus').style.display = 'block';
    showStatus('⏳ Reentrenamiento iniciado...', 'info');
    
    try {
        const response = await fetch('/api/model/retrain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs: 5 })
        });
        
        const data = await response.json();
        if (data.success) {
            showStatus('✓ Reentrenamiento completado. Nuevo modelo: ' + data.new_version, 'success');
            loadAvailableModels();
            loadCorrectionStats();
        } else {
            showStatus('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    } finally {
        document.getElementById('retrainStatus').style.display = 'none';
    }
}

// ===== TAB SWITCHING =====
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

// ===== UTILITIES =====
function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.textContent = message;
    statusDiv.className = `status-message show ${type}`;
    
    if (type !== 'info') {
        setTimeout(() => {
            statusDiv.classList.remove('show');
        }, 5000);
    }
}

console.log('Advanced script loaded');
