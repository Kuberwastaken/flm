// Artificial scores for presentation tests only. No model or corpus is loaded.
export function syntheticCoreFixture() {
  const identity = '57dcbe9b8242cc0bb897486a54139553b8093192697830993a3a76085830aaee';
  const selection = 'a'.repeat(64), cache = 'b'.repeat(64), tokenizer = 'c'.repeat(64);
  const controls = ['full','fixed_dynamics','no_lateral','no_temporal_state'];
  const shifts = [0, -.004, .03, .2];
  const runs = controls.flatMap((control, index) => [42, 43].map(seed => {
    const documents = Array.from({length:60}, (_, i) => {
      const bytes = Math.floor(1287656/60) + (i < 1287656%60 ? 1 : 0);
      const tokens = Math.floor(367981/60) + (i < 367981%60 ? 1 : 0);
      const bpb = 2 + shifts[index] + .01 * Math.sin(i + seed);
      return {document:`SYNTHETIC-ARTICLE-${i}`, bytes, tokens, nll:bpb * bytes * Math.LN2, bits_per_byte:bpb};
    });
    const nll = documents.reduce((sum, row) => sum + row.nll, 0);
    return {label:`${control}-s${seed}`, control, seed, reference:control === 'full',
      graph:'data/graphs/central-1024/graph.npz', checkpoint_step:seed === 42 ? 500 : 1000,
      checkpoint_sha256:String(index+1).repeat(64), selection_sha256:selection, test_cache_sha256:cache,
      allocated_parameters:600003, trainable_parameters:control === 'fixed_dynamics' ? 521824 : 600003,
      frozen_parameters:control === 'fixed_dynamics' ? 78179 : 0,
      score:{documents, nll, bytes:1287656, tokens:367981, bits_per_byte:nll / 1287656 / Math.LN2,
        token_perplexity:Math.exp(nll / 367981), tokenizer_sha256:tokenizer,
        mechanism:control === 'no_temporal_state' ? 'Reset both states for every token, including within chunks' : 'Carry native fast/slow state within each article; reset per article'}};
  }));
  const pair = (first, second, seed) => {
    const score = control => runs.find(row => row.control === control && row.seed === seed).score.bits_per_byte;
    const difference_bpb = score(first) - score(second);
    return {first, second, training_seed:seed, difference_bpb, lower_95:difference_bpb-.01,
      upper_95:difference_bpb+.01, replicates:10000, seed:31415, unit:'paired article resampling'};
  };
  const primary_contrasts = controls.slice(1).flatMap(control => [42, 43].map(seed => pair('full', control, seed)));
  const independent_unit_memory_contrasts = [42, 43].map(seed => pair('no_lateral', 'no_temporal_state', seed));
  const primary_means = controls.slice(1).map(control => ({control,
    mean_difference_bpb:primary_contrasts.filter(row => row.second === control).reduce((sum, row) => sum + row.difference_bpb, 0) / 2}));
  const independent_unit_memory_mean_difference_bpb = independent_unit_memory_contrasts.reduce((sum, row) => sum + row.difference_bpb, 0) / 2;
  const report = {study:'SYNTHETIC PRESENTATION FIXTURE — NOT EXPERIMENTAL RESULTS', study_identity_sha256:identity,
    selection_sha256:selection, runs, primary_contrasts, primary_means,
    independent_unit_memory_contrasts, independent_unit_memory_mean_difference_bpb};
  const release = {archive:'public/research/language-core-records.zip', archive_sha256:'d'.repeat(64),
    summary_sha256:'0'.repeat(64), bytes:300000, study_identity_sha256:identity, selection_sha256:selection,
    full_checkpoint_and_score_gate_passed:true, fresh_archive_arithmetic_verified:true,
    extraction_outside_repository:true, repository_pythonpath_removed:true, flm_and_torch_imports_disabled:true,
    arithmetic_audit:{verified_runs:8, unique_test_articles:60, scored_bytes_per_run:1287656,
      scored_tokens_per_run:367981, study_identity_sha256:identity, selection_sha256:selection,
      ...structuredClone({primary_contrasts,primary_means,independent_unit_memory_contrasts,independent_unit_memory_mean_difference_bpb})}};
  return {report, release};
}
