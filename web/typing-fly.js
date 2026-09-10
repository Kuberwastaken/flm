import './typing-fly.css';

// Presentation only. This component never reads neural state or writes to inference.
export function createTypingFly(element) {
  const caption = element.querySelector('[data-typing-caption]');
  const motion = element.querySelector('input');
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const updateMotion = () => {
    element.dataset.motion = motion.checked && !preference.matches ? 'on' : 'off';
  };
  motion.addEventListener('change', updateMotion);
  preference.addEventListener('change', updateMotion);
  updateMotion();
  return {
    setState(state) {
      element.dataset.state = state;
      caption.textContent = state === 'typing' ? 'FLM is writing' :
        state === 'reading' ? 'Reading your passage' : 'Ready to continue';
    }
  };
}
