const $ = id => document.getElementById(id);
const probability = value => Number.isFinite(value) && value >= 0 && value <= 1;
const invariantNames = ['in_degree', 'out_degree', 'incoming_sign_counts',
  'incoming_signed_weight_multiset', 'self_edges_and_weights', 'node_identities_positions_and_pools'];
const fixedStatistics = ['in_degree', 'out_degree', 'positive_in_degree', 'negative_in_degree', 'incoming_absolute_weight'];

export function validateStructureReport(report, identity) {
  if (!/^[a-f0-9]{64}$/.test(identity ?? '') || report.study_identity_sha256 !== identity) {
    throw new Error('Structural audit belongs to a different study.');
  }
  if (report.schema_version !== 1 || report.graphs?.length !== 4) throw new Error('Incomplete structural audit.');
  const seeds = [null, 101, 103, 107], hashes = new Set();
  report.graphs.forEach((graph, index) => {
    const stats = graph.statistics;
    if (graph.graph_seed !== seeds[index] || !/^[a-f0-9]{64}$/.test(graph.graph_sha256) || hashes.has(graph.graph_sha256)) {
      throw new Error('Structural audit graph identity is invalid.');
    }
    hashes.add(graph.graph_sha256);
    if (stats?.neurons !== 1024 || stats.edges !== 76130 || stats.self_edges !== 3 ||
        !probability(graph.measured_edge_overlap_fraction) || !probability(stats.reciprocal_off_diagonal_fraction) ||
        !Number.isInteger(stats.strong_components) || stats.strong_components < 1 ||
        stats.strong_component_sizes?.length !== stats.strong_components ||
        !stats.strong_component_sizes.every(n => Number.isInteger(n) && n > 0) ||
        stats.strong_component_sizes.reduce((a, b) => a + b, 0) !== stats.neurons) {
      throw new Error('Invalid structural measurements.');
    }
    if (index && (graph.accepted_swaps !== 761300 || !Number.isInteger(graph.proposed_swaps) ||
        graph.proposed_swaps < graph.accepted_swaps || graph.proposed_swaps > 7613000 ||
        invariantNames.some(name => graph.audit?.preserved?.[name] !== true) ||
        fixedStatistics.some(name => graph.audit?.maximum_absolute_node_differences?.[name] !== 0))) {
      throw new Error('A declared graph constraint did not pass.');
    }
  });
  const overlaps = report.pairwise_edge_overlap_fraction;
  if (overlaps?.length !== 4 || overlaps.some(row => row.length !== 4 || !row.every(probability))) {
    throw new Error('Incomplete graph overlap measurements.');
  }
  overlaps.forEach((row, i) => row.forEach((value, j) => {
    if (value !== overlaps[j][i] || (i === j && value !== 1) ||
        (i === 0 && value !== report.graphs[j].measured_edge_overlap_fraction)) {
      throw new Error('Inconsistent graph overlap measurements.');
    }
  }));
  return report;
}

export async function loadLanguageStructure(identity) {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/language-topology-structure.json`);
    if (!response.ok) throw new Error('The structural audit is unavailable.');
    const report = validateStructureReport(await response.json(), identity);
    const rows = report.graphs.map(graph => {
      const row = document.createElement('tr');
      const cells = [graph.graph_seed ? `Rewired · graph ${graph.graph_seed}` : 'Measured wiring',
        `${(graph.measured_edge_overlap_fraction * 100).toFixed(2)}%`,
        `${(graph.statistics.reciprocal_off_diagonal_fraction * 100).toFixed(2)}%`,
        graph.statistics.strong_components, Math.max(...graph.statistics.strong_component_sizes),
        graph.accepted_swaps?.toLocaleString() ?? '—'];
      for (const [index, value] of cells.entries()) {
        const cell = document.createElement(index === 0 ? 'th' : 'td');
        if (index === 0) cell.scope = 'row';
        cell.textContent = String(value); row.append(cell);
      }
      return row;
    });
    $('language-structure-rows').replaceChildren(...rows);
    $('language-structure-status').textContent = 'All three frozen controls pass the declared invariants. Every node has exactly the same in/out degrees, positive/negative incoming counts and incoming weight magnitudes as the measured graph. The audit also checks identities, signs, original self-edges, positions and pools.';
  } catch (error) { $('language-structure-status').textContent = error.message; }
}
