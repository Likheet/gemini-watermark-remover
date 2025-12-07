// Popup script
document.addEventListener('DOMContentLoaded', () => {
  const compressToggle = document.getElementById('compressToggle');
  const qualityRow = document.getElementById('qualityRow');
  const qualitySlider = document.getElementById('qualitySlider');
  const qualityValue = document.getElementById('qualityValue');
  const toggleRow = document.querySelector('.toggle-row');

  // Load saved settings
  chrome.storage.local.get(['compressImage', 'compressionQuality'], (result) => {
    // Default compress to false
    const isCompressed = result.compressImage === true;
    compressToggle.checked = isCompressed;
    toggleQualityRow(isCompressed);

    // Default quality to 80
    const quality = result.compressionQuality || 80;
    qualitySlider.value = quality;
    qualityValue.textContent = quality;
  });

  dashboardBtn.addEventListener('click', () => {
    chrome.tabs.create({ url: 'dashboard.html' });
  });

  compressToggle.addEventListener('change', (e) => {
    const checked = e.target.checked;
    chrome.storage.local.set({ compressImage: checked });
    toggleQualityRow(checked);
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
