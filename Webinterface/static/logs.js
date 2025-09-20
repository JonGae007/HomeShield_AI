// Logs JavaScript Funktionalität

document.addEventListener('DOMContentLoaded', function() {
    // DOM-Elemente
    const selectAllCheckbox = document.getElementById('selectAll');
    const selectAllHeaderCheckbox = document.getElementById('selectAllHeader');
    const detectionCheckboxes = document.querySelectorAll('.detection-checkbox');
    const selectedCountSpan = document.getElementById('selectedCount');
    const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
    const deleteButtons = document.querySelectorAll('.delete-btn');
    const refreshBtn = document.getElementById('refreshBtn');
    const clearAllBtn = document.getElementById('clearAllBtn');
    const deleteModal = document.getElementById('deleteModal');
    const confirmDeleteBtn = document.getElementById('confirmDelete');
    const cancelDeleteBtn = document.getElementById('cancelDelete');

    let currentDeleteAction = null;

    // Initialisierung
    updateSelectedCount();
    
    // Event Listeners

    // Alle auswählen/abwählen
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', function() {
            const isChecked = this.checked;
            detectionCheckboxes.forEach(checkbox => {
                checkbox.checked = isChecked;
            });
            if (selectAllHeaderCheckbox) {
                selectAllHeaderCheckbox.checked = isChecked;
            }
            updateSelectedCount();
        });
    }

    if (selectAllHeaderCheckbox) {
        selectAllHeaderCheckbox.addEventListener('change', function() {
            const isChecked = this.checked;
            detectionCheckboxes.forEach(checkbox => {
                checkbox.checked = isChecked;
            });
            if (selectAllCheckbox) {
                selectAllCheckbox.checked = isChecked;
            }
            updateSelectedCount();
        });
    }

    // Einzelne Checkboxen
    detectionCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            updateSelectedCount();
            
            // Prüfe ob alle ausgewählt sind
            const allSelected = Array.from(detectionCheckboxes).every(cb => cb.checked);
            const noneSelected = Array.from(detectionCheckboxes).every(cb => !cb.checked);
            
            if (selectAllCheckbox) {
                selectAllCheckbox.checked = allSelected;
                selectAllCheckbox.indeterminate = !allSelected && !noneSelected;
            }
            
            if (selectAllHeaderCheckbox) {
                selectAllHeaderCheckbox.checked = allSelected;
                selectAllHeaderCheckbox.indeterminate = !allSelected && !noneSelected;
            }
        });
    });

    // Bulk-Löschen
    if (bulkDeleteBtn) {
        bulkDeleteBtn.addEventListener('click', function() {
            const selectedIds = getSelectedIds();
            if (selectedIds.length === 0) {
                showNotification('Keine Erkennungen ausgewählt', 'warning');
                return;
            }
            
            currentDeleteAction = {
                type: 'bulk',
                ids: selectedIds
            };
            
            document.getElementById('deleteModalText').textContent = 
                `Sind Sie sicher, dass Sie ${selectedIds.length} Erkennungen löschen möchten?`;
            deleteModal.style.display = 'block';
        });
    }

    // Einzelne Lösch-Buttons
    deleteButtons.forEach(button => {
        button.addEventListener('click', function() {
            const detectionId = parseInt(this.getAttribute('data-id'));
            currentDeleteAction = {
                type: 'single',
                ids: [detectionId]
            };
            
            document.getElementById('deleteModalText').textContent = 
                'Sind Sie sicher, dass Sie diese Erkennung löschen möchten?';
            deleteModal.style.display = 'block';
        });
    });

    // Alle löschen
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', function() {
            currentDeleteAction = {
                type: 'clear-all'
            };
            
            document.getElementById('deleteModalText').textContent = 
                'Sind Sie sicher, dass Sie ALLE Erkennungen löschen möchten? Diese Aktion kann nicht rückgängig gemacht werden!';
            deleteModal.style.display = 'block';
        });
    }

    // Modal-Buttons
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', function() {
            executeDelete();
        });
    }

    if (cancelDeleteBtn) {
        cancelDeleteBtn.addEventListener('click', function() {
            closeModal();
        });
    }

    // Refresh Button
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            window.location.reload();
        });
    }

    // Modal außerhalb schließen
    deleteModal.addEventListener('click', function(e) {
        if (e.target === deleteModal) {
            closeModal();
        }
    });

    // ESC-Taste für Modal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && deleteModal.style.display === 'block') {
            closeModal();
        }
    });

    // Funktionen

    function updateSelectedCount() {
        const selectedCount = getSelectedIds().length;
        if (selectedCountSpan) {
            selectedCountSpan.textContent = `${selectedCount} ausgewählt`;
        }
        
        if (bulkDeleteBtn) {
            bulkDeleteBtn.disabled = selectedCount === 0;
        }
    }

    function getSelectedIds() {
        return Array.from(detectionCheckboxes)
            .filter(checkbox => checkbox.checked)
            .map(checkbox => parseInt(checkbox.value));
    }

    function executeDelete() {
        if (!currentDeleteAction) return;

        const loadingText = 'Lösche...';
        confirmDeleteBtn.textContent = loadingText;
        confirmDeleteBtn.disabled = true;

        if (currentDeleteAction.type === 'single') {
            deleteSingle(currentDeleteAction.ids[0]);
        } else if (currentDeleteAction.type === 'bulk') {
            deleteBulk(currentDeleteAction.ids);
        } else if (currentDeleteAction.type === 'clear-all') {
            clearAll();
        }
    }

    function deleteSingle(detectionId) {
        fetch(`/api/logs/${detectionId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            closeModal();
            if (data.success) {
                showNotification(data.message, 'success');
                removeRowFromTable(detectionId);
            } else {
                showNotification(data.error, 'error');
            }
        })
        .catch(error => {
            closeModal();
            showNotification('Fehler beim Löschen der Erkennung', 'error');
            console.error('Error:', error);
        })
        .finally(() => {
            confirmDeleteBtn.textContent = 'Löschen';
            confirmDeleteBtn.disabled = false;
        });
    }

    function deleteBulk(detectionIds) {
        fetch('/api/logs/bulk-delete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ids: detectionIds
            })
        })
        .then(response => response.json())
        .then(data => {
            closeModal();
            if (data.success) {
                showNotification(data.message, 'success');
                detectionIds.forEach(id => removeRowFromTable(id));
                
                // Checkboxen zurücksetzen
                if (selectAllCheckbox) selectAllCheckbox.checked = false;
                if (selectAllHeaderCheckbox) selectAllHeaderCheckbox.checked = false;
                updateSelectedCount();
            } else {
                showNotification(data.error, 'error');
            }
        })
        .catch(error => {
            closeModal();
            showNotification('Fehler beim Löschen der Erkennungen', 'error');
            console.error('Error:', error);
        })
        .finally(() => {
            confirmDeleteBtn.textContent = 'Löschen';
            confirmDeleteBtn.disabled = false;
        });
    }

    function clearAll() {
        fetch('/api/logs/clear-all', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                confirm: true
            })
        })
        .then(response => response.json())
        .then(data => {
            closeModal();
            if (data.success) {
                showNotification(data.message, 'success');
                // Seite neu laden nach dem Löschen aller Daten
                setTimeout(() => {
                    window.location.reload();
                }, 1000);
            } else {
                showNotification(data.error, 'error');
            }
        })
        .catch(error => {
            closeModal();
            showNotification('Fehler beim Löschen aller Erkennungen', 'error');
            console.error('Error:', error);
        })
        .finally(() => {
            confirmDeleteBtn.textContent = 'Löschen';
            confirmDeleteBtn.disabled = false;
        });
    }

    function removeRowFromTable(detectionId) {
        const row = document.querySelector(`tr[data-id="${detectionId}"]`);
        if (row) {
            row.style.opacity = '0';
            row.style.transform = 'translateX(-100%)';
            setTimeout(() => {
                row.remove();
                
                // Prüfe ob noch Zeilen vorhanden sind
                const remainingRows = document.querySelectorAll('.logs-table tbody tr');
                if (remainingRows.length === 0) {
                    // Seite neu laden wenn keine Zeilen mehr da sind
                    setTimeout(() => {
                        window.location.reload();
                    }, 500);
                }
            }, 300);
        }
    }

    function closeModal() {
        deleteModal.style.display = 'none';
        currentDeleteAction = null;
        confirmDeleteBtn.textContent = 'Löschen';
        confirmDeleteBtn.disabled = false;
    }

    function showNotification(message, type = 'info') {
        // Entferne vorhandene Notifications
        const existingNotification = document.querySelector('.notification');
        if (existingNotification) {
            existingNotification.remove();
        }

        // Erstelle neue Notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        `;

        // Füge CSS hinzu falls noch nicht vorhanden
        if (!document.querySelector('#notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 15px 20px;
                    border-radius: 8px;
                    color: white;
                    font-weight: 600;
                    z-index: 1100;
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    min-width: 300px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    animation: slideIn 0.3s ease-out;
                }
                .notification-success {
                    background-color: #28a745;
                }
                .notification-error {
                    background-color: #dc3545;
                }
                .notification-warning {
                    background-color: #ffc107;
                    color: #212529;
                }
                .notification-info {
                    background-color: #17a2b8;
                }
                .notification-close {
                    background: none;
                    border: none;
                    color: inherit;
                    font-size: 20px;
                    cursor: pointer;
                    padding: 0;
                    margin-left: auto;
                }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }

        // Füge zur Seite hinzu
        document.body.appendChild(notification);

        // Close Button Event
        notification.querySelector('.notification-close').addEventListener('click', function() {
            notification.remove();
        });

        // Auto-remove nach 5 Sekunden
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Strg+A für alle auswählen (nur wenn Focus nicht in Input-Feld)
        if (e.ctrlKey && e.key === 'a' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
            e.preventDefault();
            if (selectAllCheckbox) {
                selectAllCheckbox.checked = true;
                selectAllCheckbox.dispatchEvent(new Event('change'));
            }
        }
        
        // Delete-Taste für ausgewählte Einträge löschen
        if (e.key === 'Delete' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
            const selectedIds = getSelectedIds();
            if (selectedIds.length > 0) {
                bulkDeleteBtn.click();
            }
        }
        
        // F5 für Refresh (preventDefault um doppeltes Laden zu vermeiden)
        if (e.key === 'F5') {
            e.preventDefault();
            refreshBtn.click();
        }
    });

    // Auto-refresh alle 30 Sekunden (optional)
    const autoRefresh = localStorage.getItem('logsAutoRefresh') === 'true';
    if (autoRefresh) {
        setInterval(() => {
            // Nur refreshen wenn Modal nicht offen ist
            if (deleteModal.style.display !== 'block') {
                window.location.reload();
            }
        }, 30000);
    }

    // Speichere Scroll-Position
    const scrollPosition = sessionStorage.getItem('logsScrollPosition');
    if (scrollPosition) {
        window.scrollTo(0, parseInt(scrollPosition));
        sessionStorage.removeItem('logsScrollPosition');
    }

    window.addEventListener('beforeunload', function() {
        sessionStorage.setItem('logsScrollPosition', window.scrollY.toString());
    });
});