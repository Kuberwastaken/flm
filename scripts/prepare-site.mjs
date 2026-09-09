import { mkdir, copyFile, readdir, readFile, writeFile } from 'node:fs/promises';
await mkdir('public/licenses', { recursive: true });
await mkdir('public/data', { recursive: true });
for (const [source, target] of [['data/cards/ami.json', 'public/data/ami.json'], ['data/graphs/central-1024/graph-card.json', 'public/data/graph-card.json']])
  await copyFile(source, target);
const files = (await readdir('licenses')).filter(name => /\.(txt|md)$/.test(name));
for (const name of files) await copyFile(`licenses/${name}`, `public/licenses/${name}`);
await copyFile('LICENSE', 'public/licenses/FLM-MIT.txt');
files.unshift('FLM-MIT.txt');
const escape = text => text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
await writeFile('public/licenses/index.html', `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FLM — Attribution and licenses</title><style>body{max-width:760px;margin:40px auto;padding:0 24px;background:#faf8f5;color:#292820;font:17px/1.7 Georgia,serif}a{color:#a74c20}h1{font-size:30px}li{margin:12px 0}</style><a href="/">← ChatFLM</a><h1>Attribution and licenses</h1><p>FLM's original code is MIT licensed. MaleCNS brain data and AMI meeting transcripts use CC BY 4.0. The NeuroMechFly body and imported kinematics retain their upstream notices. These sources describe different specimens and components.</p><ul>${files.map(name => `<li><a href="${encodeURIComponent(name)}">${escape(name)}</a></li>`).join('')}</ul><p>The complete model and data provenance is available in the <a href="/#research">research notebook</a>.</p></html>\n`);
console.log('Prepared dataset cards and component attribution.');
