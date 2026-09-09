/** Framework-free inference and a separate, resettable local readout adapter. */
export class FLM {
  constructor(config, binary) {
    if (config.format !== 'flm-browser-v1' || binary.byteLength !== config.weights_bytes)
      throw new Error('Unsupported or incomplete model package.');
    this.config = config;
    this.n = config.neurons;
    this.features = config.features;
    this.vocab = config.vocabulary;
    this.arrays = {};
    for (const [name, spec] of Object.entries(config.arrays)) {
      if (!Number.isInteger(spec.offset) || !Number.isInteger(spec.length) || spec.offset < 0 || spec.length < 0 ||
          spec.offset % 4 || spec.offset + spec.length * 4 > binary.byteLength)
        throw new Error(`Invalid array ${name}.`);
      const Type = spec.dtype === 'uint32' ? Uint32Array : Float32Array;
      this.arrays[name] = new Type(binary, spec.offset, spec.length);
    }
    const a = this.arrays;
    if (a.offsets.length !== this.n + 1 || a.offsets[0] !== 0 || a.offsets[this.n] !== a.weights.length ||
        a.weights.length !== a.sources.length || a.drives.length !== this.vocab * this.n ||
        a.readout_weight.length !== this.vocab * this.features || a.pool.length !== this.n)
      throw new Error('The model graph has inconsistent dimensions.');
    for (let i = 0; i < this.n; i++) {
      if (a.offsets[i] > a.offsets[i + 1] || a.pool[i] >= config.pools) throw new Error('Invalid graph indices.');
    }
    for (const source of a.sources) if (source >= this.n) throw new Error('Invalid source neuron.');
    this.adapter = new Float32Array(this.vocab * this.features);
    this.adapterBias = new Float32Array(this.vocab);
    this.disabled = new Uint8Array(this.n);
    this.adaptationEnabled = true;
    this.recurrenceEnabled = true;
    this.reset();
  }

  reset() {
    this.h = new Float32Array(this.n);
    this.slow = new Float32Array(this.n);
    this.next = new Float32Array(this.n);
    this.encoded = new Float32Array(this.features);
    this.logits = new Float32Array(this.vocab);
  }

  step(token) {
    if (!Number.isInteger(token) || token < 0 || token >= this.vocab) throw new Error('Invalid input token.');
    const a = this.arrays, n = this.n, pools = this.config.pools;
    this.encoded.fill(0);
    for (let j = 0; j < n; j++) {
      let drive = a.drives[token * n + j];
      if (this.recurrenceEnabled && this.config.variant !== 'no_recurrence') {
        let incoming = 0;
        for (let e = a.offsets[j]; e < a.offsets[j + 1]; e++) incoming += a.weights[e] * this.h[a.sources[e]];
        drive += incoming;
      }
      const value = this.disabled[j] ? 0 : (1 - a.alpha[j]) * this.h[j] + a.alpha[j] * Math.tanh(drive);
      this.next[j] = value;
      this.slow[j] = this.disabled[j] || this.config.variant === 'no_slow' ? 0 :
        (1 - a.beta[j]) * this.slow[j] + a.beta[j] * this.next[j];
      this.encoded[a.pool[j]] += this.next[j] / a.pool_sizes[a.pool[j]];
      this.encoded[pools + a.pool[j]] += this.slow[j] / a.pool_sizes[a.pool[j]];
    }
    [this.h, this.next] = [this.next, this.h];
    let mean = 0, variance = 0;
    for (const value of this.encoded) mean += value / this.features;
    for (const value of this.encoded) variance += (value - mean) ** 2 / this.features;
    const inverse = 1 / Math.sqrt(variance + this.config.norm_epsilon);
    for (let k = 0; k < this.features; k++)
      this.encoded[k] = (this.encoded[k] - mean) * inverse * a.norm_weight[k] + a.norm_bias[k];
    for (let v = 0; v < this.vocab; v++) {
      let score = a.readout_bias[v] + (this.adaptationEnabled ? this.adapterBias[v] : 0);
      const start = v * this.features;
      for (let k = 0; k < this.features; k++)
        score += (a.readout_weight[start + k] + (this.adaptationEnabled ? this.adapter[start + k] : 0)) * this.encoded[k];
      this.logits[v] = score;
    }
    return this.logits;
  }

  learn(target, rate = 0.03, decay = 0.00001) {
    if (!Number.isInteger(target) || target < 0 || target >= this.vocab || rate <= 0 || rate > 1)
      throw new Error('Invalid learning parameters.');
    const probabilities = softmax(this.logits);
    const loss = -Math.log(Math.max(probabilities[target], 1e-30));
    const step = rate / Math.sqrt(this.features);
    for (let v = 0; v < this.vocab; v++) {
      const error = probabilities[v] - Number(v === target);
      this.adapterBias[v] = (1 - decay) * this.adapterBias[v] - step * error;
      for (let k = 0; k < this.features; k++) {
        const i = v * this.features + k;
        this.adapter[i] = Math.max(-2, Math.min(2, (1 - decay) * this.adapter[i] - step * error * this.encoded[k]));
      }
    }
    return loss;
  }

  clearLearning() { this.adapter.fill(0); this.adapterBias.fill(0); }

  exportLearning() {
    return { format: 'flm-adapter-v1', modelHash: this.config.weights_sha256,
      weights: Array.from(this.adapter), bias: Array.from(this.adapterBias) };
  }

  importLearning(value) {
    if (value?.format !== 'flm-adapter-v1' || value.modelHash !== this.config.weights_sha256 ||
        !Array.isArray(value.weights) || value.weights.length !== this.adapter.length ||
        !Array.isArray(value.bias) || value.bias.length !== this.adapterBias.length ||
        !value.weights.every(x => Number.isFinite(x) && Math.abs(x) <= 2) ||
        !value.bias.every(x => Number.isFinite(x) && Math.abs(x) <= 100))
      throw new Error('This learning file is invalid or belongs to another model.');
    this.adapter.set(value.weights); this.adapterBias.set(value.bias);
  }
}

export function softmax(logits, temperature = 1) {
  const out = new Float64Array(logits.length);
  let max = -Infinity;
  for (const x of logits) max = Math.max(max, x / temperature);
  let sum = 0;
  for (let i = 0; i < out.length; i++) { out[i] = Math.exp(logits[i] / temperature - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function random(seed = 42) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
}

export function sample(logits, rng, { temperature = 0.8, topK = 40 } = {}) {
  if (!Number.isFinite(temperature) || temperature < 0.05 || temperature > 2 || !Number.isInteger(topK) || topK < 1)
    throw new Error('Invalid sampling settings.');
  const candidates = Array.from(logits, (score, token) => ({ score, token }))
    .filter(x => x.token !== 256 && (x.token === 257 || x.token === 10 || x.token === 9 || x.token >= 32))
    .sort((a, b) => b.score - a.score).slice(0, Math.min(topK, logits.length));
  const probabilities = softmax(candidates.map(x => x.score), temperature);
  let draw = rng();
  for (let i = 0; i < candidates.length; i++) { draw -= probabilities[i]; if (draw <= 0) return candidates[i].token; }
  return candidates.at(-1).token;
}
