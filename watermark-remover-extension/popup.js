// Popup script
document.addEventListener('DOMContentLoaded', () => {
  const compressToggle = document.getElementById('compressToggle');
  const preloadToggle = document.getElementById('preloadToggle');
  const qualityRow = document.getElementById('qualityRow');
  const qualitySlider = document.getElementById('qualitySlider');
  const qualityValue = document.getElementById('qualityValue');
  const toggleRow = document.querySelector('.toggle-row');

  // Load saved settings
  chrome.storage.local.get(['compressImage', 'compressionQuality', 'preloadModel'], (result) => {
    // Default compress to false
    const isCompressed = result.compressImage === true;
    compressToggle.checked = isCompressed;
    toggleQualityRow(isCompressed);

    // Default quality to 80
    const quality = result.compressionQuality || 80;
    qualitySlider.value = quality;
    qualityValue.textContent = quality;

    // Default preload to false (off by default for potato PCs)
    preloadToggle.checked = result.preloadModel === true;
  });

  dashboardBtn.addEventListener('click', () => {
    chrome.tabs.create({ url: 'dashboard.html' });
  });

  compressToggle.addEventListener('change', (e) => {
    const checked = e.target.checked;
    chrome.storage.local.set({ compressImage: checked });
    toggleQualityRow(checked);
  });

  preloadToggle.addEventListener('change', (e) => {
    chrome.storage.local.set({ preloadModel: e.target.checked });
  });

  qualitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    qualityValue.textContent = val;
    chrome.storage.local.set({ compressionQuality: parseInt(val) });
  });

  function toggleQualityRow(show) {
    if (show) {
      qualityRow.style.display = 'flex';
      toggleRow.classList.add('expanded');
    } else {
      qualityRow.style.display = 'none';
      toggleRow.classList.remove('expanded');
    }
  }
});
