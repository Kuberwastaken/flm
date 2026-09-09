// Execute the unmodified web worker protocol without a browser or DOM.
import { parentPort } from 'node:worker_threads';
import { readFile } from 'node:fs/promises';
globalThis.self = globalThis;
globalThis.postMessage = message => parentPort.postMessage(message);
globalThis.fetch = async path => {
  if (!/^\/models\/[a-z-]+\/[a-z.-]+$/.test(path)) throw new Error('Unexpected test asset path');
  try { return new Response(await readFile(new URL(`../../public${path}`, import.meta.url))); }
  catch { return new Response('', {status: 404}); }
};
await import('../../web/worker.js');
parentPort.on('message', data => self.onmessage({data}));
parentPort.postMessage({type: 'harness-ready'});
