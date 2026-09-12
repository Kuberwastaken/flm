import catalog from './model-catalog.json' with { type: 'json' };
export const DEFAULT_MODEL = catalog.default_flm;
export const MODEL_PACKAGES = Object.freeze({
  ...Object.fromEntries(Object.entries(catalog.models).sort((a,b)=>(a[0]===DEFAULT_MODEL?-1:b[0]===DEFAULT_MODEL?1:0)).map(([key,value])=>[key,Object.freeze(value)])),
  ami: Object.freeze({ path: 'flm-compact', lexical: false, architecture:'flm', label:'FLM · AMI meetings', name:'FLM' }),
  wikitext: Object.freeze({ path: 'flm-wikitext', lexical: true, architecture:'flm', label:'FLM · WikiText-2 · original link', hidden:true }),
  babylm: Object.freeze({ path: 'flm-babylm', lexical: true, architecture:'flm', label:'FLM · BabyLM 10M · original preview', hidden:true }),
});
