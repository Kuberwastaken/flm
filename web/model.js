/** Framework-free inference and a separate, resettable local readout adapter. */
function packFloats(values) {
  const bytes = new Uint8Array(values.length * 4), view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat32(i * 4, values[i], true);
  let text = '';
  for (let i = 0; i < bytes.length; i += 16384) text += String.fromCharCode(...bytes.subarray(i, i + 16384));
  return btoa(text);
}
function unpackFloats(text, length) {
  if (typeof text !== 'string' || text.length !== Math.ceil(length * 4 / 3) * 4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(text))
    throw new Error('Invalid packed learning data.');
  const decoded = atob(text);
  if (decoded.length !== length * 4) throw new Error('Invalid packed learning dimensions.');
  const bytes = Uint8Array.from(decoded, c => c.charCodeAt(0)), view = new DataView(bytes.buffer);
  return Float32Array.from({length}, (_, i) => view.getFloat32(i * 4, true));
}

export class FLM {
  constructor(config, binary) {
    if (!['flm-browser-v1', 'flm-browser-v2'].includes(config.format) || binary.byteLength !== config.weights_bytes)
      throw new Error('Unsupported or incomplete model package.');
    this.config = config;
    this.n = config.neurons;
    this.features = config.features;
    this.vocab = config.vocabulary;
    this.lexical = config.format === 'flm-browser-v2';
    this.pooledFeatures = config.pools * 2;
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
        a.weights.length !== a.sources.length || a.pool.length !== this.n)
      throw new Error('The model graph has inconsistent dimensions.');
    const expected = { alpha: this.n, beta: this.n, pool_sizes: config.pools,
      norm_weight: this.pooledFeatures, norm_bias: this.pooledFeatures, readout_bias: this.vocab,
      ...(this.lexical ? { embedding: this.vocab * this.features, input_weight: this.n * this.features,
        input_bias: this.n, projection_weight: this.features * this.pooledFeatures, projection_bias: this.features }
        : { drives: this.vocab * this.n, readout_weight: this.vocab * this.features }) };
    for (const [name, length] of Object.entries(expected))
      if (a[name]?.length !== length) throw new Error(`Invalid dimensions for ${name}.`);
    if ((this.lexical && config.embedding !== this.features) || (!this.lexical && this.features !== this.pooledFeatures))
      throw new Error('Inconsistent feature dimensions.');
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
    this.pooled = this.lexical ? new Float32Array(this.pooledFeatures) : this.encoded;
    this.logits = new Float32Array(this.vocab);
  }

  step(token) {
    if (!Number.isInteger(token) || token < 0 || token >= this.vocab) throw new Error('Invalid input token.');
    const a = this.arrays, n = this.n, pools = this.config.pools;
    this.pooled.fill(0);
    for (let j = 0; j < n; j++) {
      let drive;
      if (this.lexical) {
        drive = a.input_bias[j];
        for (let k = 0; k < this.features; k++) drive += a.input_weight[j * this.features + k] * a.embedding[token * this.features + k];
      } else drive = a.drives[token * n + j];
      if (this.recurrenceEnabled && this.config.variant !== 'no_recurrence') {
        let incoming = 0;
        for (let e = a.offsets[j]; e < a.offsets[j + 1]; e++) incoming += a.weights[e] * this.h[a.sources[e]];
        drive += incoming;
      }
      const value = this.disabled[j] ? 0 : (1 - a.alpha[j]) * this.h[j] + a.alpha[j] * Math.tanh(drive);
      this.next[j] = value;
      this.slow[j] = this.disabled[j] || this.config.variant === 'no_slow' ? 0 :
        (1 - a.beta[j]) * this.slow[j] + a.beta[j] * this.next[j];
      this.pooled[a.pool[j]] += this.next[j] / a.pool_sizes[a.pool[j]];
      this.pooled[pools + a.pool[j]] += this.slow[j] / a.pool_sizes[a.pool[j]];
    }
    [this.h, this.next] = [this.next, this.h];
    let mean = 0, variance = 0;
    for (const value of this.pooled) mean += value / this.pooledFeatures;
    for (const value of this.pooled) variance += (value - mean) ** 2 / this.pooledFeatures;
    const inverse = 1 / Math.sqrt(variance + this.config.norm_epsilon);
    for (let k = 0; k < this.pooledFeatures; k++)
      this.pooled[k] = (this.pooled[k] - mean) * inverse * a.norm_weight[k] + a.norm_bias[k];
    if (this.lexical) for (let k = 0; k < this.features; k++) {
      let value = a.projection_bias[k];
      for (let j = 0; j < this.pooledFeatures; j++) value += a.projection_weight[k * this.pooledFeatures + j] * this.pooled[j];
      this.encoded[k] = value;
    }
    const readout = this.lexical ? a.embedding : a.readout_weight;
    for (let v = 0; v < this.vocab; v++) {
      let score = a.readout_bias[v] + (this.adaptationEnabled ? this.adapterBias[v] : 0);
      const start = v * this.features;
      for (let k = 0; k < this.features; k++)
        score += (readout[start + k] + (this.adaptationEnabled ? this.adapter[start + k] : 0)) * this.encoded[k];
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
    return { format: 'flm-adapter-v2', dtype: 'float32-le', modelHash: this.config.weights_sha256,
      weights: packFloats(this.adapter), bias: packFloats(this.adapterBias) };
  }

  importLearning(value) {
    if (!['flm-adapter-v1', 'flm-adapter-v2'].includes(value?.format) || value.modelHash !== this.config.weights_sha256)
      throw new Error('This learning file is invalid or belongs to another model.');
    let weights, bias;
    if (value.format === 'flm-adapter-v2') {
      if (value.dtype !== 'float32-le') throw new Error('Unknown learning number format.');
      weights = unpackFloats(value.weights, this.adapter.length); bias = unpackFloats(value.bias, this.adapterBias.length);
    } else {
      if (!Array.isArray(value.weights) || !Array.isArray(value.bias)) throw new Error('Invalid learning arrays.');
      weights = value.weights; bias = value.bias;
    }
    if (weights.length !== this.adapter.length || bias.length !== this.adapterBias.length ||
        !weights.every(x => Number.isFinite(x) && Math.abs(x) <= 2) ||
        !bias.every(x => Number.isFinite(x) && Math.abs(x) <= 100))
      throw new Error('This learning file is invalid or belongs to another model.');
    this.adapter.set(weights); this.adapterBias.set(bias);
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

export function sample(logits, rng, { temperature = 0.8, topK = 40, allowed = null } = {}) {
  if (!Number.isFinite(temperature) || temperature < 0.05 || temperature > 2 || !Number.isInteger(topK) || topK < 1)
    throw new Error('Invalid sampling settings.');
  const candidates = Array.from(logits, (score, token) => ({ score, token }))
    .filter(x => allowed ? allowed[x.token] : x.token !== 256 && (x.token === 257 || x.token === 10 || x.token === 9 || x.token >= 32))
    .sort((a, b) => b.score - a.score).slice(0, Math.min(topK, logits.length));
  if (!candidates.length) throw new Error('No allowed sampling candidates.');
  const probabilities = softmax(candidates.map(x => x.score), temperature);
  let draw = rng();
  for (let i = 0; i < candidates.length; i++) { draw -= probabilities[i]; if (draw <= 0) return candidates[i].token; }
  return candidates.at(-1).token;
}
