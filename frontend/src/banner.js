export function clearErrorBanner() {
  document.querySelector('.error-banner')?.remove();
}

export function showErrorBanner(message) {
  clearErrorBanner();

  const banner = document.createElement('div');
  banner.className = 'error-banner';
  banner.setAttribute('role', 'alert');
  banner.innerHTML = `
    <div class="error-banner-content">
      <span class="error-banner-text"></span>
      <button type="button" class="btn btn-secondary">Dismiss</button>
    </div>
  `;
  banner.querySelector('.error-banner-text').textContent = message;
  banner.querySelector('button').addEventListener('click', () => banner.remove());
  document.body.prepend(banner);
  setTimeout(() => banner.remove(), 15000);
}
