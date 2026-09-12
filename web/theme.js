export function initializeTheme() {
  const root = document.documentElement, media = matchMedia('(prefers-color-scheme: dark)');
  let stored;
  try { stored = localStorage.getItem('chatflm-theme'); } catch {}
  let theme = ['light', 'dark'].includes(stored) ? stored : media.matches ? 'dark' : 'light';
  const apply = () => {
    root.dataset.theme = theme;
    const button = document.getElementById('theme-toggle');
    button.textContent = theme === 'dark' ? 'Light' : 'Dark';
    button.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
    document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#191919' : '#faf8f5';
    document.dispatchEvent(new Event('flm-theme'));
  };
  document.getElementById('theme-toggle').onclick = () => {
    theme = theme === 'dark' ? 'light' : 'dark'; stored = theme;
    try { localStorage.setItem('chatflm-theme', theme); } catch {}
    apply();
  };
  media.addEventListener('change', () => { if (!stored) { theme = media.matches ? 'dark' : 'light'; apply(); } });
  apply();
}
