// Display delimiting only: these base models have no trained role hierarchy.
export function chatOutput(text, final = false) {
  const labels = ['\nUser:', '\nSystem:', '\nAssistant:'];
  let end = text.length, stopped = false;
  for (const label of labels) {
    const index = text.indexOf(label);
    if (index >= 0 && index < end) { end = index; stopped = true; }
  }
  let visible = text.slice(0, end);
  // Hold a partial turn marker during streaming so it never flashes as an answer.
  if (!stopped && !final) for (let n = 1; n <= Math.min(visible.length, 10); n++) {
    if (labels.some(label => label.startsWith(visible.slice(-n)))) { visible = visible.slice(0, -n); break; }
  }
  return { text: visible, stopped };
}
