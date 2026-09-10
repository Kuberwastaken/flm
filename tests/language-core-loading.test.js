import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fetchCoreRecords} from '../web/language-core.js';
import {syntheticCoreFixture} from './helpers/core-results-fixture.js';

function fixture() {
  const {report, release} = syntheticCoreFixture();
  const bytes = new TextEncoder().encode(JSON.stringify(report));
  release.summary_sha256 = createHash('sha256').update(bytes).digest('hex');
  return {report, release, bytes};
}

test('fetches a checksum-verified complete pair without cached mixed release files', async () => {
  const {report, release, bytes} = fixture(), calls = [];
  const controller = new AbortController();
  const result = await fetchCoreRecords('/preview', {signal:controller.signal, fetcher:async (url, options) => {
    calls.push({url, options});
    return new Response(url.endsWith('results.json') ? bytes : JSON.stringify(release));
  }});
  assert.deepEqual(result, {report, release, base:'/preview/'});
  assert.deepEqual(calls.map(row => row.url), ['/preview/research/language-core-results.json', '/preview/research/language-core-release.json']);
  assert.ok(calls.every(row => row.options.cache === 'no-store' && row.options.signal === controller.signal));
});

test('either missing resource refuses before reading response bodies', async () => {
  for (const missing of ['results.json', 'release.json']) {
    let bodies = 0;
    await assert.rejects(fetchCoreRecords('/', {fetcher:async url => ({
      ok:!url.endsWith(missing), status:url.endsWith(missing) ? 404 : 200,
      arrayBuffer:async () => { bodies++; }, json:async () => { bodies++; }
    })}), /unavailable \(404\)/);
    assert.equal(bodies, 0);
  }
});

test('changed summary bytes or a different frozen release cannot return results', async () => {
  for (const mismatch of ['bytes', 'selection']) {
    const {release, bytes} = fixture();
    if (mismatch === 'bytes') bytes[bytes.length - 2] ^= 1;
    else release.selection_sha256 = 'f'.repeat(64);
    await assert.rejects(fetchCoreRecords('/', {fetcher:async url =>
      new Response(url.endsWith('results.json') ? bytes : JSON.stringify(release))}),
    mismatch === 'bytes' ? /summary bytes match release/ : /displayed selection/);
  }
});

test('network and caller cancellation failures propagate without a partial result', async () => {
  for (const error of [new TypeError('Network failure'), new DOMException('Cancelled', 'AbortError')]) {
    await assert.rejects(fetchCoreRecords('/', {fetcher:async () => { throw error; }}), actual => actual === error);
  }
});
