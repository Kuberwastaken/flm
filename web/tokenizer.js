/** Lossless GPT-2 byte pretokenization + the project's train-only BPE merges. */
export class ByteBPE {
  constructor(specification) {
    if (specification.format !== 'flm-byte-bpe-v1' || specification.byte_ids?.length !== 256 ||
        specification.pieces?.length !== specification.vocabulary) throw new Error('Invalid tokenizer package.');
    this.config = specification; this.pieces = specification.pieces.map(piece => Uint8Array.from(piece));
    this.ranks = new Map(); this.cache = new Map(); this.encoder = new TextEncoder();
    for (const [rank, [left, right, token]] of specification.merges.entries()) {
      if (![left, right, token].every(x => Number.isInteger(x) && x >= 2 && x < this.pieces.length))
        throw new Error('Invalid tokenizer merge.');
      this.ranks.set(`${left},${right}`, { rank, token });
    }
    this.pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  }

  encode(text, boundaries = false) {
    const output = boundaries ? [this.config.bos] : [];
    for (const match of text.matchAll(this.pattern)) {
      const piece = match[0]; let tokens = this.cache.get(piece);
      if (!tokens) {
        tokens = Array.from(this.encoder.encode(piece), byte => this.config.byte_ids[byte]);
        while (tokens.length > 1) {
          let best = Infinity, replacement;
          for (let i = 0; i < tokens.length - 1; i++) {
            const merge = this.ranks.get(`${tokens[i]},${tokens[i + 1]}`);
            if (merge && merge.rank < best) { best = merge.rank; replacement = merge.token; }
          }
          if (!Number.isFinite(best)) break;
          const next = [];
          for (let i = 0; i < tokens.length; i++) {
            const merge = i + 1 < tokens.length ? this.ranks.get(`${tokens[i]},${tokens[i + 1]}`) : null;
            if (merge?.rank === best) { next.push(replacement); i++; } else next.push(tokens[i]);
          }
          tokens = next;
        }
        if (this.cache.size >= 10000) this.cache.clear();
        this.cache.set(piece, tokens);
      }
      for (const token of tokens) output.push(token);
    }
    if (boundaries) output.push(this.config.eos);
    return output;
  }

  decode(tokens) {
    const bytes = new Uint8Array(tokens.reduce((sum, token) => sum + this.pieces[token].length, 0));
    let offset = 0;
    for (const token of tokens) { bytes.set(this.pieces[token], offset); offset += this.pieces[token].length; }
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  }
}
