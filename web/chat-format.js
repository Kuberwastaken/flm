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

// A chat presentation guard, not a model score or a change to sampling probabilities.
export function repetitionLoop(text) {
  let start = Infinity, kind;
  const found = (index, reason) => { if (index < start) { start = index; kind = reason; } };
  for (const match of text.matchAll(/([\p{L}\p{N}])\1{11,}/gu)) found(match.index, 'repeated-character');
  for (const match of text.matchAll(/(.{1,8}?)\1{7,}/gsu)) {
    if (match[0].length >= 32 && /[\p{L}\p{N}]/u.test(match[1])) found(match.index, 'repeated-pattern');
  }
  // Catch near-identical B...A...B loops without treating ordinary prose as repetition.
  for (let end = 24; end <= text.length; end++) {
    const window = text.slice(end - 24, end), counts = new Map();
    for (const char of window) if (/[\p{L}\p{N}]/u.test(char)) counts.set(char, (counts.get(char) || 0) + 1);
    if ([...counts.values()].some(count => count >= 22)) found(end - 24, 'dominant-character');
    if (end >= 48) {
      const digits = text.slice(end - 48, end);
      if (/^\d{48}$/.test(digits)) {
        const frequencies = Array.from({length: 10}, (_, n) => [...digits].filter(char => char === String(n)).length).sort((a,b) => b-a);
        if (frequencies[0] + frequencies[1] >= 40) found(end - 48, 'low-variety-digits');
      }
    }
  }
  return Number.isFinite(start) ? { start, kind } : null;
}
