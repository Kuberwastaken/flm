import { FLM } from './model.js';

const sigmoid = x => 1 / (1 + Math.exp(-x));
// Abramowitz-Stegun erf approximation; parity is checked against PyTorch GELU.
function erf(x) {
  const sign = Math.sign(x), v = Math.abs(x), t = 1 / (1 + .3275911 * v);
  return sign * (1 - (((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-v*v));
}

export class BrowserBaseline {
  constructor(config, binary) {
    this.config=config; this.c=config.architecture_config;
    if (config.format!=='flm-baseline-browser-v1' || binary.byteLength!==config.weights_bytes ||
        !['gru','transformer'].includes(config.architecture) || !this.c.tied_readout || this.c.embedding!==96 ||
        this.c.vocabulary!==4096 || this.c.hidden!==200 || this.c.width!==108 || this.c.heads!==6 || this.c.layers!==2 || this.c.window!==96)
      throw new Error('Unsupported baseline package.');
    this.architecture=config.architecture; this.arrays={};
    for (const [name,spec] of Object.entries(config.arrays)) {
      if (spec.dtype!=='float32' || spec.offset%4 || spec.length!==spec.shape.reduce((a,b)=>a*b,1) || spec.offset+spec.length*4>binary.byteLength)
        throw new Error('Invalid baseline array: '+name);
      const values=new Float32Array(binary,spec.offset,spec.length);
      if (!values.every(Number.isFinite)) throw new Error('Nonfinite baseline weights.');
      this.arrays[name]=values;
    }
    this.vocab=config.vocabulary; this.features=config.features; this.n=0;
    this.adapter=new Float32Array(this.vocab*this.features); this.adapterBias=new Float32Array(this.vocab);
    this.adaptationEnabled=true; this.recurrenceEnabled=true; this.disabled=new Uint8Array(0); this.reset();
  }
  reset() {
    this.h=new Float32Array(0); this.slow=new Float32Array(0);
    this.hidden=new Float32Array(this.c.hidden); this.cache=Array.from({length:this.c.layers},()=>[]); this.position=0;
    this.encoded=new Float32Array(this.features); this.logits=new Float32Array(this.vocab);
  }
  linear(x,name) {
    const w=this.arrays[name+'.weight'], b=this.arrays[name+'.bias'];
    if (!w || !b || w.length!==b.length*x.length) throw new Error('Invalid linear dimensions: '+name);
    const y=new Float32Array(b.length);
    for(let i=0;i<y.length;i++) { let v=b[i]; for(let j=0;j<x.length;j++) v+=w[i*x.length+j]*x[j]; y[i]=v; }
    return y;
  }
  norm(x,name) {
    const w=this.arrays[name+'.weight'],b=this.arrays[name+'.bias'];
    if(w?.length!==x.length || b?.length!==x.length) throw new Error('Invalid normalization dimensions.');
    const mean=x.reduce((a,b)=>a+b,0)/x.length, variance=x.reduce((a,b)=>a+(b-mean)**2,0)/x.length;
    return Float32Array.from(x,(v,i)=>(v-mean)/Math.sqrt(variance+1e-5)*w[i]+b[i]);
  }
  gru(x) {
    const a=this.arrays, h=this.hidden, n=h.length, input=new Float32Array(n*3), recurrent=new Float32Array(n*3);
    for(let i=0;i<3*n;i++) {
      let v=a['core.bias_ih_l0'][i], r=a['core.bias_hh_l0'][i];
      for(let j=0;j<x.length;j++) v+=a['core.weight_ih_l0'][i*x.length+j]*x[j];
      for(let j=0;j<n;j++) r+=a['core.weight_hh_l0'][i*n+j]*h[j];
      input[i]=v; recurrent[i]=r;
    }
    for(let i=0;i<n;i++) {
      const r=sigmoid(input[i]+recurrent[i]), z=sigmoid(input[n+i]+recurrent[n+i]);
      const next=Math.tanh(input[2*n+i]+r*recurrent[2*n+i]); h[i]=(1-z)*next+z*h[i];
    }
    return this.linear(h,'readout');
  }
  transformer(x) {
    let values=this.linear(x,'input_projection'); const c=this.c, d=c.width/c.heads;
    for(let layer=0;layer<c.layers;layer++) {
      const prefix=`blocks.${layer}`, qkv=this.linear(this.norm(values,prefix+'.norm1'),prefix+'.qkv');
      const q=qkv.slice(0,c.width), k=qkv.slice(c.width,2*c.width), v=qkv.slice(2*c.width);
      for(let head=0;head<c.heads;head++) for(let j=0;j<d;j+=2) {
        const angle=Math.fround(this.position*Math.fround(10000**(-j/d))), cs=Math.cos(angle), sn=Math.sin(angle), i=head*d+j;
        for(const tensor of [q,k]) { const a=tensor[i], b=tensor[i+1]; tensor[i]=a*cs-b*sn; tensor[i+1]=a*sn+b*cs; }
      }
      const cache=this.cache[layer]; cache.push({k,v}); if(cache.length>c.window) cache.shift();
      const attended=new Float32Array(c.width);
      for(let head=0;head<c.heads;head++) {
        const scores=cache.map(entry=>{ let sum=0; for(let j=0;j<d;j++) sum+=q[head*d+j]*entry.k[head*d+j]; return sum/Math.sqrt(d); });
        const max=Math.max(...scores), exp=scores.map(v=>Math.exp(v-max)), total=exp.reduce((a,b)=>a+b,0);
        for(let j=0;j<d;j++) { let sum=0; for(let t=0;t<cache.length;t++) sum+=exp[t]/total*cache[t].v[head*d+j]; attended[head*d+j]=sum; }
      }
      const output=this.linear(attended,prefix+'.output'); values=Float32Array.from(values,(v,i)=>v+output[i]);
      const hidden=this.linear(this.norm(values,prefix+'.norm2'),prefix+'.ff.0');
      for(let i=0;i<hidden.length;i++) hidden[i]=.5*hidden[i]*(1+erf(hidden[i]/Math.SQRT2));
      const ff=this.linear(hidden,prefix+'.ff.2'); values=Float32Array.from(values,(v,i)=>v+ff[i]);
      // The next step receives at most 95 past positions, matching the native cache.
      if(cache.length===c.window) cache.shift();
    }
    this.position++; return this.linear(this.norm(values,'norm'),'readout');
  }
  step(token) {
    if(!Number.isInteger(token)||token<0||token>=this.vocab) throw new Error('Invalid input token.');
    const embedding=this.arrays['embedding.weight'];
    if(embedding?.length!==this.vocab*this.features) throw new Error('Invalid embedding dimensions.');
    const x=embedding.subarray(token*this.features,(token+1)*this.features);
    this.encoded=this.architecture==='gru'?this.gru(x):this.transformer(x);
    for(let v=0;v<this.vocab;v++) {
      let score=this.arrays.output_bias[v]+(this.adaptationEnabled?this.adapterBias[v]:0);
      for(let j=0;j<this.features;j++) score+=(embedding[v*this.features+j]+(this.adaptationEnabled?this.adapter[v*this.features+j]:0))*this.encoded[j];
      this.logits[v]=score;
    }
    return this.logits;
  }
}
for (const method of ['learn','clearLearning','exportLearning','importLearning']) BrowserBaseline.prototype[method]=FLM.prototype[method];
