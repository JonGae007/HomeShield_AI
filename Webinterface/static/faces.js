// Gesichterverwaltung JavaScript

// Modal Variablen
let addFaceModal = null;
let currentMethod = 'upload';

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    addFaceModal = document.getElementById('addFaceModal');
    
    // Click outside modal to close
    addFaceModal.addEventListener('click', function(event) {
        if (event.target === addFaceModal) {
            closeAddFaceModal();
        }
    });
    
    // Escape key to close modal
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && addFaceModal.style.display === 'block') {
            closeAddFaceModal();
        }
    });
    
    // File input change handler
    const fileInput = document.getElementById('faceImage');
    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            previewImage(event.target);
        });
    }
    
    // Camera selector change handler
    const cameraSelect = document.getElementById('cameraSelect');
    if (cameraSelect) {
        cameraSelect.addEventListener('change', function() {
            if (this.value) {
                loadCameraPreview(this.value);
            }
        });
    }
});

// Modal Funktionen
function openAddFaceModal() {
    if (addFaceModal) {
        addFaceModal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        // Reset form
        resetForm();
        
        // Set default method
        selectMethod('upload');
    }
}

function closeAddFaceModal() {
    if (addFaceModal) {
        addFaceModal.style.display = 'none';
        document.body.style.overflow = 'auto';
        
        // Reset form
        resetForm();
    }
}

function resetForm() {
    const form = document.getElementById('addFaceForm');
    if (form) {
        form.reset();
    }
    
    // Clear image preview
    const preview = document.getElementById('imagePreview');
    if (preview) {
        preview.innerHTML = '';
    }
    
    // Hide camera preview
    const cameraPreview = document.getElementById('cameraPreview');
    const captureBtn = document.getElementById('captureBtn');
    if (cameraPreview) {
        cameraPreview.style.display = 'none';
        cameraPreview.src = '';
    }
    if (captureBtn) {
        captureBtn.style.display = 'none';
    }
}

// Method Selection
function selectMethod(method) {
    currentMethod = method;
    
    // Update buttons
    const uploadBtn = document.getElementById('uploadBtn');
    const cameraBtn = document.getElementById('cameraBtn');
    
    if (uploadBtn && cameraBtn) {
        uploadBtn.classList.toggle('active', method === 'upload');
        cameraBtn.classList.toggle('active', method === 'camera');
    }
    
    // Show/hide method content
    const uploadMethod = document.getElementById('uploadMethod');
    const cameraMethod = document.getElementById('cameraMethod');
    
    if (uploadMethod && cameraMethod) {
        uploadMethod.classList.toggle('active', method === 'upload');
        cameraMethod.classList.toggle('active', method === 'camera');
    }
    
    // Update form action based on method
    const form = document.getElementById('addFaceForm');
    if (form) {
        if (method === 'camera') {
            // Remove file input requirement for camera method
            const fileInput = document.getElementById('faceImage');
            if (fileInput) {
                fileInput.removeAttribute('required');
            }
        } else {
            // Add file input requirement for upload method
            const fileInput = document.getElementById('faceImage');
            if (fileInput) {
                fileInput.setAttribute('required', 'required');
            }
        }
    }
}

// Image Preview
function previewImage(input) {
    const preview = document.getElementById('imagePreview');
    if (!preview) return;
    
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Vorschau">`;
        };
        
        reader.readAsDataURL(input.files[0]);
    } else {
        preview.innerHTML = '';
    }
}

// Camera Functions
function loadCameraPreview(cameraId) {
    const preview = document.getElementById('cameraPreview');
    const captureBtn = document.getElementById('captureBtn');
    
    if (!preview || !captureBtn) return;
    
    // Show loading state
    preview.src = '/static/icons/camera.png';
    preview.style.display = 'block';
    captureBtn.style.display = 'none';
    
    // Load camera stream (placeholder - in real implementation would connect to camera)
    setTimeout(() => {
        // Simulate camera feed loading
        preview.src = `/api/camera/${cameraId}/preview?t=${Date.now()}`;
        captureBtn.style.display = 'block';
        
        // Handle image load error
        preview.onerror = function() {
            preview.src = '/static/icons/camera.png';
            showNotification('Kamera-Vorschau nicht verfügbar', 'error');
        };
    }, 1000);
}

function capturePhoto() {
    const cameraSelect = document.getElementById('cameraSelect');
    const preview = document.getElementById('cameraPreview');
    
    if (!cameraSelect.value) {
        showNotification('Bitte wählen Sie eine Kamera aus', 'error');
        return;
    }
    
    // Foto von der Kamera aufnehmen
    showNotification('Foto wird aufgenommen...', 'info');
    
    fetch(`/api/camera/${cameraSelect.value}/capture`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Verstecktes Input-Feld für die Bilddaten erstellen/aktualisieren
            let hiddenInput = document.getElementById('cameraPhotoData');
            if (!hiddenInput) {
                hiddenInput = document.createElement('input');
                hiddenInput.type = 'hidden';
                hiddenInput.id = 'cameraPhotoData';
                hiddenInput.name = 'camera_photo_data';
                document.getElementById('addFaceForm').appendChild(hiddenInput);
            }
            
            // Base64-Bilddaten speichern
            hiddenInput.value = data.image_data;
            
            // Kamera-ID für das Backend speichern
            let cameraIdInput = document.getElementById('cameraIdInput');
            if (!cameraIdInput) {
                cameraIdInput = document.createElement('input');
                cameraIdInput.type = 'hidden';
                cameraIdInput.id = 'cameraIdInput';
                cameraIdInput.name = 'camera_id';
                document.getElementById('addFaceForm').appendChild(cameraIdInput);
            }
            cameraIdInput.value = cameraSelect.value;
            
            showNotification('Foto erfolgreich aufgenommen!', 'success');
            
            // Optionale Vorschau des aufgenommenen Fotos
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                // Vorschau aktualisieren
                preview.src = canvas.toDataURL('image/jpeg');
            };
            
            img.src = 'data:image/jpeg;base64,' + data.image_data;
            
        } else {
            showNotification(data.error || 'Fehler beim Aufnehmen des Fotos', 'error');
        }
    })
    .catch(error => {
        console.error('Fehler beim Foto aufnehmen:', error);
        showNotification('Fehler beim Aufnehmen des Fotos', 'error');
    });
}

// Face Management Functions
function editFace(faceId, faceName) {
    const newName = prompt(`Neuen Namen für "${faceName}" eingeben:`, faceName);
    
    if (newName && newName.trim() && newName.trim() !== faceName) {
        // In real implementation would call API to update name
        fetch(`/api/faces/${faceId}/update`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: newName.trim()
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(`Name erfolgreich zu "${newName.trim()}" geändert`, 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                showNotification(data.message || 'Fehler beim Ändern des Namens', 'error');
            }
        })
        .catch(error => {
            showNotification('Fehler beim Ändern des Namens', 'error');
        });
    }
}

function deleteFace(faceId, faceName) {
    if (confirm(`Möchten Sie das Gesicht "${faceName}" wirklich löschen?`)) {
        fetch(`/delete_face/${faceId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(data.message, 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                showNotification(data.message || 'Fehler beim Löschen', 'error');
            }
        })
        .catch(error => {
            showNotification('Fehler beim Löschen des Gesichts', 'error');
        });
    }
}

// Utility Functions
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        background: ${type === 'error' ? 'rgba(229, 62, 62, 0.9)' : 
                    type === 'success' ? 'rgba(72, 187, 120, 0.9)' : 
                    'rgba(59, 130, 246, 0.9)'};
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transform: translateX(400px);
        transition: transform 0.3s ease;
        max-width: 400px;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 4 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Form Validation
document.addEventListener('submit', function(event) {
    const form = event.target;
    
    if (form.id === 'addFaceForm') {
        const name = form.querySelector('#faceName').value.trim();
        
        if (!name) {
            event.preventDefault();
            showNotification('Bitte geben Sie einen Namen ein', 'error');
            return false;
        }
        
        if (currentMethod === 'upload') {
            const fileInput = form.querySelector('#faceImage');
            if (!fileInput.files || !fileInput.files[0]) {
                event.preventDefault();
                showNotification('Bitte wählen Sie ein Bild aus', 'error');
                return false;
            }
        } else if (currentMethod === 'camera') {
            const cameraSelect = form.querySelector('#cameraSelect');
            const cameraPhotoData = form.querySelector('#cameraPhotoData');
            
            if (!cameraSelect.value) {
                event.preventDefault();
                showNotification('Bitte wählen Sie eine Kamera aus', 'error');
                return false;
            }
            
            if (!cameraPhotoData || !cameraPhotoData.value) {
                event.preventDefault();
                showNotification('Bitte nehmen Sie zuerst ein Foto auf', 'error');
                return false;
            }
        }
        
        // Show loading state
        const submitBtn = form.querySelector('#submitBtn');
        if (submitBtn) {
            submitBtn.textContent = 'Wird hinzugefügt...';
            submitBtn.disabled = true;
        }
    }
});

// Drag and Drop für File Upload
function initDragAndDrop() {
    const dropZone = document.querySelector('.file-upload-label');
    if (!dropZone) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-hover');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-hover');
        });
    });
    
    dropZone.addEventListener('drop', function(e) {
        const files = e.dataTransfer.files;
        const fileInput = document.getElementById('faceImage');
        
        if (files.length > 0 && fileInput) {
            fileInput.files = files;
            previewImage(fileInput);
        }
    });
}

// Initialize drag and drop when modal opens
function openAddFaceModal() {
    if (addFaceModal) {
        addFaceModal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        resetForm();
        selectMethod('upload');
        initDragAndDrop();
    }
}