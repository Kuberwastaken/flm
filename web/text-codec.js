import { ByteBPE } from './tokenizer.js';

/** Boundary IDs and byte accounting shared by generation, scoring and adaptation. */
export class TextCodec {
  constructor(config, tokenizer = null) {
    this.bpe = tokenizer ? new ByteBPE(tokenizer) : null;
    if ((config.format === 'flm-browser-v2') !== Boolean(this.bpe)) throw new Error('Missing or unexpected tokenizer.');
    if (this.bpe && (tokenizer.tokenizer_sha256 !== config.tokenizer_sha256 || tokenizer.vocabulary !== config.vocabulary))
      throw new Error('Tokenizer does not belong to this model.');
    this.bos = config.bos ?? 256; this.eos = config.eos ?? 257;
    this.encoder = new TextEncoder();
    this.pieces = this.bpe?.pieces ?? Array.from({length: 258}, (_, i) => Uint8Array.from(i < 256 ? [i] : []));
    this.allowed = Uint8Array.from(this.pieces, (bytes, i) => Number(i === this.eos ||
      (i !== this.bos && bytes.length > 0 && bytes.every(b => b === 9 || b === 10 || b >= 32))));
  }
  encode(text) { return this.bpe ? this.bpe.encode(text) : Array.from(this.encoder.encode(text)); }
  bytes(token) { return this.pieces[token]; }
  label(token) {
    if (token === this.eos) return 'end';
    if (token === this.bos) return 'start';
    const bytes = this.bytes(token);
    try {
      return new TextDecoder('utf-8', {fatal: true}).decode(bytes).replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥');
    } catch { return Array.from(bytes, b => `\\x${b.toString(16).padStart(2, '0')}`).join(''); }
  }
}
