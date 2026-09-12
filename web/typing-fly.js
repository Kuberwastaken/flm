import './typing-fly.css';

// Presentation only. This component never reads neural state or writes to inference.
export function createTypingFly(element) {
  const caption = element.querySelector('[data-typing-caption]');
  const scene = element.querySelector('.typing-fly-scene');
  const loading = element.querySelector('[data-typing-loading]');
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  let view, pending, frame = 0, visible = false, failed = false, state = 'idle';
  let elapsed = 0, previous = 0;
  const tick = timestamp => {
    frame = requestAnimationFrame(tick);
    if (timestamp - previous < 1000 / 30) return;
    elapsed += Math.min(0.05, (timestamp - previous) / 1000); previous = timestamp;
    view.draw(elapsed, true);
  };
  const updateMotion = () => {
    const animate = !!view?.rig && !failed && visible && !document.hidden && !preference.matches && state === 'typing';
    element.dataset.motion = animate ? 'on' : 'off';
    if (animate && !frame) { previous = performance.now(); frame = requestAnimationFrame(tick); }
    else if (!animate) {
      cancelAnimationFrame(frame); frame = 0;
      if (view && !failed && visible && !document.hidden) view.draw(0, false);
    }
  };
  const unavailable = () => {
    failed = true; updateMotion(); view?.dispose(); view = undefined;
    loading.hidden = false; loading.textContent = '3D illustration unavailable';
  };
  scene.addEventListener('webglcontextlost', unavailable, true);
  const observer = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (visible && !pending) pending = import('./typing-fly-view.js').then(async ({ TypingFlyView }) => {
      view = new TypingFlyView(scene); await view.load(); loading.hidden = true; updateMotion();
    }).catch(unavailable);
    updateMotion();
  });
  observer.observe(element);
  preference.addEventListener('change', updateMotion);
  document.addEventListener('visibilitychange', updateMotion);
  return {
    setState(next) {
      if (state === next) return;
      state = next;
      element.dataset.state = state;
      caption.textContent = state === 'typing' ? 'Writing' :
        state === 'reading' ? 'Reading your message' : 'Ready';
      updateMotion();
    }
  };
}
