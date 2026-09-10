import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateStructureReport } from '../web/language-structure.js';

const report = JSON.parse(await readFile(new URL('../public/research/language-topology-structure.json', import.meta.url), 'utf8'));

test('structural display uses all four actual graphs from the frozen study', () => {
  assert.equal(validateStructureReport(report, report.study_identity_sha256), report);
  assert.deepEqual(report.graphs.map(g => g.graph_seed), [null, 101, 103, 107]);
  assert.throws(() => validateStructureReport(report, '0'.repeat(64)), /different study/);
});

test('structural display refuses incomplete constraints and inconsistent graph statistics', () => {
  for (const mutate of [
    r => r.graphs.pop(),
    r => { r.graphs[1].graph_seed = 107; },
    r => { r.graphs[1].graph_sha256 = r.graphs[0].graph_sha256; },
    r => { r.graphs[1].audit.preserved.out_degree = false; },
    r => { r.graphs[1].audit.maximum_absolute_node_differences.negative_in_degree = 1; },
    r => { r.graphs[1].statistics.reciprocal_off_diagonal_fraction = NaN; },
    r => { r.graphs[1].statistics.strong_component_sizes[0] -= 1; },
    r => { r.graphs[1].accepted_swaps = 0; },
    r => { r.pairwise_edge_overlap_fraction[1][2] += .01; },
    r => { r.graphs[1].measured_edge_overlap_fraction = 1; }
  ]) {
    const changed = structuredClone(report); mutate(changed);
    assert.throws(() => validateStructureReport(changed, report.study_identity_sha256));
  }
});
