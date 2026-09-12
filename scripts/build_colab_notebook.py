"""Build the auditable Colab notebook from readable cells; no executed outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap


def build(revision):
    cells = []
    def cell(kind, text):
        text = textwrap.dedent(text).strip() + '\n'
        value = dict(cell_type=kind, id=hashlib.sha256(text.encode()).hexdigest()[:12],
                     metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code': value.update(execution_count=None, outputs=[])
        cells.append(value)
    def md(text): cell('markdown', text)
    def code(text): cell('code', text)

    md('''
    # FLM on a T4: full-graph capacity and learning probe
    **Kuber Mehta Â· FLM / ChatFLM**

    Train a fly-derived recurrent language model from random weights. Target **150M or 300M parameters**
    using **all 166,700 acquired neurons**, or run the existing 1,024-neuron graph as a small smoke check.
    There is no pretrained transformer inside FLM. A separate random transformer control is selectable.

    This is an engineering notebook, **not an already trained larger ChatFLM release**.
    Local CPU checks passed; T4 fit, throughput and language quality still need a GPU run.
    Budget: 40 minutes inside the trainer, with an optional three-update profile mode.
    Setup/downloads are additional; the soft deadline cannot interrupt an active update or disk write.

    Adapted workflow ideas from the audited [Gemma/SmolLM notebooks](https://github.com/Kuberwastaken/flm/blob/main/docs/COLAB-NOTEBOOK-AUDIT.md):
    real hardware reporting, assistant-only loss, accumulation, checkpointing and explicit export.
    The model, graph/data pipeline and training loop below are FLM implementations.
    The frozen Mac study and anatomical stopping rule remain separate and unchanged.

    **Colab: Runtime â†’ Change runtime type â†’ T4 GPU. Then run cells in order.**
    Free-tier allocation is not guaranteed. Drive mounting is optional but recommended for resumable runs.
    No Hub upload, API key, paid runtime or publication is triggered by this notebook.
    ''')
    code('''
    import time, sys, subprocess, json, os
    from pathlib import Path
    os.environ['USE_TF'] = '0'
    NOTEBOOK_STARTED = time.monotonic()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before running the capacity probe. CPU smoke tests live in the repo.")
    print(torch.__version__, torch.cuda.get_device_name(0))
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
    # Keep Colab's installed CUDA torch. Do not install a CPU wheel over it.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'datasets==4.3.0', 'tokenizers==0.22.2', 'transformers==4.56.2',
                    'scipy>=1.13', 'requests>=2.31', 'matplotlib>=3.8'], check=True)
    ''')
    code(f'''
    REPO_REVISION = {revision!r}
    ROOT = Path('/content/flm-colab')
    if not ROOT.exists():
        ROOT.mkdir()
        subprocess.run(['git', 'init', str(ROOT)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(ROOT), 'remote', 'add', 'origin', 'https://github.com/Kuberwastaken/flm.git'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth', '1', 'origin', REPO_REVISION], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'checkout', '--detach', REPO_REVISION], check=True)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    print('Pinned code:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    ''')
    md('''
    ## Choose one run

    `full-150m` and `full-300m` keep the same acquired neurons and modeled signed edges;
    lexical width changes to approach the parameter target. This increases learned language interfaces,
    not the number of cells. The graph has **24,469,412 nonzero-sign edges**; **1,113,526** zero-sign
    outgoing edges are omitted. About 96.85% of acquired contacts remain in fast computation.
    Full acquired coverage does not recover unknown signs, physiology or all biological connectivity.

    `smoke` uses the already published 1,024-neuron graph and a roughly 0.6M model.
    Choose `transformer` for an independently initialized Llama-style dense control on identical prepared data.
    Compare matched **scored-token counts** as well as wall time; a 40-minute run can process different amounts.
    This control is not an exact reproduction of pretrained SmolLM or Gemma.

    Start with `pretrain`. A later `chat_sft` run must load your own foundation checkpoint.
    The conversation phase teaches assistant boundaries; it does not supply missing foundational knowledge.
    ''')
    code('''
    PRESET = 'full-150m'  # 'full-150m', 'full-300m', 'smoke'
    ARCHITECTURE = 'flm'  # 'flm' or 'transformer'
    PHASE = 'pretrain'  # 'pretrain' or 'chat_sft'
    PROFILE_ONLY = False  # True stops after three actual optimizer updates; False continues within the budget.
    USE_DRIVE = True
    RESUME = False
    RUN_NAME = 'flm-full-150m-profile-01'  # Change for each distinct experiment.
    INITIAL_CHECKPOINT = ''  # For chat_sft: your own prior run's /last.pt
    BUDGET_SECONDS = 40 * 60
    LENGTH = 128
    MICRO_BATCH = 1
    ACCUMULATION = 4
    CHUNK = 8
    SPARSE_BACKEND = 'csr'  # Explicit 'edge' fallback; select a new run name if changed.
    DATA_VERSION = 'v1'  # Change after changing preprocessing or an incomplete preparation.
    MAX_DOCUMENTS = 10000
    MAX_TRAIN_TOKENS = 4_000_000
    SEED = 42
    assert PRESET in ('full-150m', 'full-300m', 'smoke')
    assert ARCHITECTURE in ('flm', 'transformer')
    assert PHASE in ('pretrain', 'chat_sft')
    if USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        STORAGE = Path('/content/drive/MyDrive/FLM-Colab')
    else:
        STORAGE = Path('/content/flm-artifacts')
        print('Local runtime storage is temporary. Download the checkpoint before disconnecting.')
    STORAGE.mkdir(parents=True, exist_ok=True)
    OUTPUT = STORAGE / 'runs' / RUN_NAME
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    ''')
    md('''
    ## Acquire data and anatomy with recorded identities

    Foundation data: a bounded sample of **FineWeb-Edu**, already educationally filtered web text.
    Chat data: **OASST1**, reviewed English messages with non-synthetic flags and rank-0 assistant replies.
    Conversation trees remain together; assistant content and end-of-turn tokens receive the chat loss.
    Corpora have exact-text deduplication and fixed hash partitions, not a complete benchmark contamination audit.

    The BPE vocabulary was fitted previously on BabyLM training text. Seven explicit special tokens create a
    separate **4,101-ID** vocabulary. This notebook does not reuse the legacy browser runtime's ID offset.
    Data revisions, document/tree identities, hashes and exposure counts are saved. Raw text/tokens remain local.
    ''')
    code('''
    from flm.colab_data import prepare, load_prepared
    from flm.colab_capacity import full_graph, load_graph, sized_config, CapacityFLM, environment, train
    from flm.provenance import sha256, write_json
    from tokenizers import Tokenizer
    KIND = 'fineweb' if PHASE == 'pretrain' else 'oasst'
    DATA = STORAGE / 'data' / f'{KIND}-{DATA_VERSION}-l{LENGTH}-d{MAX_DOCUMENTS}-t{MAX_TRAIN_TOKENS}'
    if not (DATA / 'data-card.json').exists():
        prepare(ROOT, DATA, kind=KIND, length=LENGTH, max_documents=MAX_DOCUMENTS, max_train_tokens=MAX_TRAIN_TOKENS)
    examples, validation, data_card = load_prepared(DATA)
    tokenizer = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    print(json.dumps({k: v for k, v in data_card.items() if k != 'files'}, indent=2))
    graph = None
    graph_card = None
    if ARCHITECTURE == 'flm':
        if PRESET == 'smoke':
            GRAPH_DIR = ROOT / 'data/graphs/central-1024'
        else:
            GRAPH_DIR = STORAGE / 'data/full-acquired-graph-v1'
            if not (GRAPH_DIR / 'graph-card.json').exists():
                from flm.acquire_graph import acquire, extract
                cache, source = ROOT / 'work/connectome-cache', ROOT / 'work/connectome-arrays'
                pinned = acquire(ROOT / 'data/source-manifest.json', cache)
                extract(cache, source, pinned)
                full_graph(source, GRAPH_DIR)
        graph_card = json.loads((GRAPH_DIR / 'graph-card.json').read_text())
        assert sha256(GRAPH_DIR / 'graph.npz') == graph_card['graph_sha256']
        graph = load_graph(GRAPH_DIR / 'graph.npz')
        print({k: graph_card.get(k) for k in ['neurons', 'edges', 'retained_contacts', 'scope']})
    ''')
    md('''
    ## Initialize, then measure a complete update

    Sparse routing and fast/slow state use float32; dense input/readout matrices use CUDA float16 autocast.
    Checkpointed unrolling preserves gradients through the configured window while recomputing blocks.
    CSR avoids the original COO dense weight-gradient temporary. `edge` implements a chunked first-order fallback.
    Neither path has a measured T4 speed claim yet; backend support is checked by the actual optimizer probe.

    All graph edge gains, time constants and lexical interfaces are trainable. Parameter groups are printed
    so a larger lexical interface cannot be mistaken for a larger biological mechanism.
    ''')
    code('''
    from dataclasses import asdict
    from flm.colab_reference import TransformerReference, reference_config
    TARGET = {'smoke': 0.6, 'full-150m': 150, 'full-300m': 300}[PRESET]
    identity = dict(name=RUN_NAME, architecture=ARCHITECTURE, phase=PHASE, seed=SEED,
                    code_revision=REPO_REVISION, data_card_sha256=sha256(DATA / 'data-card.json'),
                    tokenizer_sha256=sha256(DATA / 'tokenizer.json'),
                    graph_sha256=graph_card['graph_sha256'] if graph_card else None,
                    initialized_from='random' if PHASE == 'pretrain' else 'own_foundation_checkpoint')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if ARCHITECTURE == 'flm':
        config = sized_config(graph, tokenizer.get_vocab_size(), TARGET)
        model = CapacityFLM(graph, config, chunk=CHUNK, sparse_backend=SPARSE_BACKEND)
    else:
        config = reference_config(tokenizer.get_vocab_size(), TARGET)
        model = TransformerReference(config)
    if PHASE == 'chat_sft' and not RESUME:
        initial = Path(INITIAL_CHECKPOINT)
        assert initial.is_file(), 'chat_sft requires your own completed foundation checkpoint'
        receipt = json.loads(initial.with_suffix('.sha256.json').read_text())
        assert sha256(initial) == receipt['sha256'], 'Initial checkpoint checksum mismatch'
        saved = torch.load(initial, map_location='cpu', weights_only=False)
        assert saved['identity']['model'] == asdict(config), 'Model configuration mismatch'
        for field in ('graph_sha256', 'tokenizer_sha256', 'architecture'):
            assert saved['identity']['experiment'][field] == identity[field], field
        model.load_state_dict(saved['model'])
        identity['initial_checkpoint_sha256'] = receipt['sha256']
        del saved
    elif PHASE == 'chat_sft' and RESUME:
        # Preserve the foundation lineage in the run's own identity.
        identity = json.loads((OUTPUT / 'identity.json').read_text())
    write_json(OUTPUT / 'identity.json', identity)
    print(json.dumps(model.parameter_card(), indent=2))
    try:
        model = model.to('cuda')
        report = train(model, examples, validation, output=OUTPUT, identity=identity,
                       pad=data_card['pad'], seconds=BUDGET_SECONDS,
                       max_steps=3 if PROFILE_ONLY else 100000,
                       batch_size=MICRO_BATCH, accumulation=ACCUMULATION, length=LENGTH,
                       lr=3e-4 if PHASE == 'pretrain' else 5e-5, resume=RESUME)
    except (RuntimeError, FloatingPointError) as error:
        write_json(OUTPUT / 'failure.json', dict(status='probe_failed', error=str(error), environment=environment(),
                   identity=identity, config=asdict(config), elapsed_seconds=time.monotonic()-NOTEBOOK_STARTED))
        raise  # Keep the failure. No silent graph reduction or hidden transformer substitution.
    print({k: v for k, v in report.items() if k not in ('updates', 'binding', 'parameters')})
    ''')
    md('''
    ## Read the result before extending the run

    `gpu_fit_verified` requires completed optimizer updates. The first updates include warmup and Adam state allocation.
    The development panel is only two deterministic windows: a quick learning check, not evidence of general chat quality.
    Test data are integrity-hashed but never scored here. Inspect token counts, nonfinite failures, memory and throughput.

    After a successful profile, set `PROFILE_ONLY=False` and `RESUME=True`, keeping the run name and numerical settings.
    This resumes the model, optimizer, scaler, RNG and data cursor. To compare architectures, use a separate run name and
    record identical corpus/token exposure; the default wall-clock budget alone does not isolate architecture.
    Stop this scale if three updates fail, gradients are nonfinite, or measured throughput makes the desired budget impractical.
    A valid negative anatomical gate from the frozen Mac study remains a negative result regardless of this pilot.
    ''')
    code('''
    import matplotlib.pyplot as plt
    updates = report['updates']
    if updates:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout='constrained')
        axes[0].plot([r['step'] for r in updates], [r['loss'] for r in updates])
        axes[0].set(xlabel='Optimizer update', ylabel='Training nats / scored token', title='Actual update loss')
        axes[1].plot([r['step'] for r in updates], [r['tokens_per_second'] for r in updates])
        axes[1].set(xlabel='Optimizer update', ylabel='Scored tokens / second', title='Complete update throughput')
        fig.savefig(OUTPUT / 'learning-and-throughput.png', dpi=180)
        plt.show()
        measured = sum(r['scored_tokens'] for r in updates) / sum(r['seconds'] for r in updates)
        print(f'Observed update throughput: {measured:.1f} scored tokens/s')
        print(f'At this short-run rate, 10M additional tokens: {10_000_000/measured/3600:.1f} hours, excluding saving/evaluation.')
    ''')
    md('''
    ## Inspect unedited output and actual state

    The foundation checkpoint is a next-token model. Use the chat formatting only after `chat_sft`.
    The state plot below records computed neuron states; it is not a scripted animation or a behavioral task result.
    A 3-update checkpoint should not be expected to generate useful language.
    ''')
    code('''
    PROMPT = 'A fruit fly can learn'
    MAX_NEW_TOKENS = 32
    if ARCHITECTURE == 'flm':
        if PHASE == 'chat_sft':
            prompt_ids = [data_card['bos'], tokenizer.token_to_id('<|user|>')]
            prompt_ids += tokenizer.encode(PROMPT, add_special_tokens=False).ids
            prompt_ids += [tokenizer.token_to_id('<|end|>'), tokenizer.token_to_id('<|assistant|>')]
        else:
            prompt_ids = [data_card['bos']] + tokenizer.encode(PROMPT, add_special_tokens=False).ids
        model.eval()
        result, states = [], []
        sampled_neurons = torch.linspace(0, model.config.neurons-1, min(256, model.config.neurons), device='cuda').long()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            logits, state = model(torch.tensor([prompt_ids], device='cuda'))
            for _ in range(MAX_NEW_TOKENS):
                top_values, top_ids = logits[0, -1].float().topk(40)
                token = int(top_ids[torch.multinomial((top_values/.8).softmax(-1), 1)])
                result.append(token)
                logits, state = model(torch.tensor([[token]], device='cuda'), state)
                states.append(state[0][0, sampled_neurons].float().cpu().numpy())
                if token in (data_card['eos'], tokenizer.token_to_id('<|end|>')):
                    break
        generated = tokenizer.decode(result, skip_special_tokens=False)
        print(PROMPT + generated)
        write_json(OUTPUT / 'sample.json', dict(prompt=PROMPT, token_ids=result, text=generated,
                   temperature=.8, top_k=40, seed=SEED, repetition_guard=False))
        import numpy as np
        np.savez_compressed(OUTPUT / 'sample-activity.npz', fast=np.asarray(states), neuron_index=sampled_neurons.cpu().numpy())
        fig, ax = plt.subplots(figsize=(9, 3), layout='constrained')
        ax.imshow(np.asarray(states).T, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        ax.set(xlabel='Generated token', ylabel='Uniformly sampled neuron index', title='Actual fast state at every generated token')
        fig.savefig(OUTPUT / 'actual-neuron-state.png', dpi=180)
        plt.show()
    ''')
    md('''
    ## Keep the artifacts

    `last.pt` contains model, optimizer, RNG and data position. Keep its SHA256 receipt, `identity.json`,
    `report.json`, prepared tokenizer/data card and graph card together. Checkpoints are your own training artifacts;
    do not load pickle checkpoints from unknown sources. Drive files remain in your account; nothing is published.

    Large checkpoints require a separate browser runtime/export review. This notebook does **not** replace the live
    ChatFLM catalog or claim that the existing JavaScript runtime supports a 150M/full-graph download.
    For practical pretrained references, see the audited SmolLM/Gemma notebook links above. Their external pretraining
    must be disclosed separately from a matched, randomly initialized control.
    ''')
    code('''
    print('Artifacts:', OUTPUT)
    print('Notebook elapsed seconds:', round(time.monotonic() - NOTEBOOK_STARTED, 1))
    print('Checkpoint SHA256:', json.loads((OUTPUT / 'last.sha256.json').read_text())['sha256'])
    print('No upload or model promotion was performed.')
    ''')
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python'), accelerator='GPU',
                              colab=dict(name='FLM_T4_Capacity.ipynb', provenance=[])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Use a full immutable Git commit')
    output = Path('notebooks/FLM_T4_Capacity.ipynb'); output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(build(args.revision), indent=1, ensure_ascii=False)+'\n', encoding='utf8')
