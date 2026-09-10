"""Acquire and audit canonical SCAN splits; never fit or score a neural model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from .provenance import sha256, write_json

REVISION = 'c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f'
BASE = f'https://raw.githubusercontent.com/brendenlake/SCAN/{REVISION}/'
FILES = {
    'LICENSE': (1532, '5caff9408d400e1d9b271bf48b967154cdf6a37c16c965f91775da3af9291f67'),
    'README.md': (5429, 'a383edc2fcc023202b12cafcf937754a681637afbb4ac930ac577d2fa2036c53'),
    'tasks.txt': (4080388, 'd824c7d9e9149ba497e25f701981dac79954e04bf670d22e5c209f076ac6c89f'),
    'simple_split/tasks_train_simple.txt': (3267938, '941bb8a088c5f53ceff12fde902dc008933cf0c4203cc672c47b5f79d73262dd'),
    'simple_split/tasks_test_simple.txt': (812450, '1fe1c8f5a19d0dc40e41bab94278f45f66f3dc415a978bd29855a23440610fc6'),
    'length_split/tasks_train_length.txt': (2723418, 'b034212dc7cf87ce62f3a067dd6ad49c4d5cd18d51d615d13825516bcb1ebbc2'),
    'length_split/tasks_test_length.txt': (1356970, '0760752ed34a75116ae11dcc1ee17526b7500060da1366540730fa24894d7db3'),
    'add_prim_split/tasks_train_addprim_jump.txt': (2579619, 'e3c78485692e028eed4205e1b8b3fd0e785ef2c12d45b5b89204dadf5b239e23'),
    'add_prim_split/tasks_test_addprim_jump.txt': (1531555, '8147ae2651422c701162e174d9d66158daa57858a26b30317ac61797118ac7a5'),
}
SPLITS = {
    'simple': ('simple_split/tasks_train_simple.txt', 'simple_split/tasks_test_simple.txt'),
    'length': ('length_split/tasks_train_length.txt', 'length_split/tasks_test_length.txt'),
    'add_primitive_jump': ('add_prim_split/tasks_train_addprim_jump.txt', 'add_prim_split/tasks_test_addprim_jump.txt'),
}
ACTIONS = ('I_WALK', 'I_LOOK', 'I_RUN', 'I_JUMP', 'I_TURN_LEFT', 'I_TURN_RIGHT')


def interpret(command):
    """Independent grammar oracle for source audit, NEVER a model prediction.

    Lake and Baroni (2018), supplementary figures 1 and 2 define these rules.
    No learned state or benchmark prediction is consulted.
    """
    if not isinstance(command, str) or not command or command != ' '.join(command.split()):
        raise ValueError('Expected a nonempty canonical SCAN command')

    def simple(text):
        words = text.split(' '); repeat = 1
        if words[-1] in ('twice', 'thrice'):
            repeat = 2 if words.pop() == 'twice' else 3
        if len(words) == 1 and words[0] in ('walk', 'look', 'run', 'jump'):
            return ('I_' + words[0].upper(),) * repeat
        match = re.fullmatch(r'(walk|look|run|jump|turn)(?: (opposite|around))? (left|right)', ' '.join(words))
        if match is None:
            raise ValueError('Command is outside the SCAN grammar: ' + text)
        verb, modifier, direction = match.groups()
        turn = ('I_TURN_' + direction.upper(),)
        action = () if verb == 'turn' else ('I_' + verb.upper(),)
        if modifier == 'around':
            result = (turn + action) * 4
        elif modifier == 'opposite':
            result = turn * 2 + action
        else:
            result = turn + action
        return result * repeat

    conjunctions = [word for word in command.split() if word in ('and', 'after')]
    if not conjunctions:
        return simple(command)
    if len(conjunctions) != 1:
        raise ValueError('SCAN permits only one top-level conjunction')
    conjunction = conjunctions[0]
    left, right = command.split(' ' + conjunction + ' ')
    first, second = simple(left), simple(right)
    return first + second if conjunction == 'and' else second + first


def parse(payload, source):
    text = payload.decode('utf8', errors='strict')
    rows = []
    for index, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r'IN: (.+) OUT: (I_[A-Z_]+(?: I_[A-Z_]+)*)', line)
        if match is None:
            raise ValueError(f'Malformed SCAN source row {source}:{index}')
        command, target = match.groups(); actions = tuple(target.split())
        if actions != interpret(command):
            raise ValueError(f'Independent interpretation disagrees at {source}:{index}')
        rows.append(dict(id=hashlib.sha256(command.encode('utf8')).hexdigest(),
                         source=source, source_line=index, command=command, actions=list(actions)))
    if not rows:
        raise ValueError('SCAN partition is empty')
    return rows


def acquire(raw, name):
    size, digest = FILES[name]; path = raw / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with urlopen(Request(BASE + name, headers={'User-Agent': 'FLM-data-acquisition'}), timeout=30) as response:
                with temporary.open('wb') as handle:
                    written = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        written += len(chunk)
                        if written > size:
                            raise ValueError('SCAN download exceeds its pinned size')
                        handle.write(chunk)
            if temporary.stat().st_size != size or sha256(temporary) != digest:
                raise ValueError('SCAN download failed its pinned checksum')
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError('Local SCAN source changed: ' + name)
    return path


def statistics(rows):
    counts = Counter(row['command'] for row in rows)
    actions = Counter(action for row in rows for action in row['actions'])
    return dict(rows=len(rows), unique_commands=len(counts), repeated_rows=len(rows)-len(counts),
                maximum_command_multiplicity=max(counts.values()),
                command_words=sum(len(row['command'].split()) for row in rows),
                actions=sum(actions.values()), action_counts=dict(sorted(actions.items())),
                action_length_histogram=dict(sorted(Counter(len(row['actions']) for row in rows).items())),
                command_length_histogram=dict(sorted(Counter(len(row['command'].split()) for row in rows).items())),
                walk_turn_only=sum(set(row['actions']) <= {'I_WALK', 'I_TURN_LEFT', 'I_TURN_RIGHT'} for row in rows))


def audit_pair(training, test, canonical, split):
    train_commands = {row['command'] for row in training}
    test_commands = {row['command'] for row in test}
    if train_commands & test_commands:
        raise ValueError('A command crosses the official train/test boundary')
    if train_commands | test_commands != set(canonical):
        raise ValueError('Official split does not cover the complete command universe')
    for row in training + test:
        if row['actions'] != canonical[row['command']]:
            raise ValueError('Split target differs from the canonical source')
    if split == 'length':
        if max(len(row['actions']) for row in training) > 22 or min(len(row['actions']) for row in test) <= 22:
            raise ValueError('Official action-length boundary changed')
    if split == 'add_primitive_jump':
        if not any(row['command'] == 'jump' for row in training):
            raise ValueError('The primitive jump example is missing')
        if any('jump' in row['command'].split() and row['command'] != 'jump' for row in training):
            raise ValueError('Compositional jump command leaked into training')
        if any('jump' not in row['command'].split() or row['command'] == 'jump' for row in test):
            raise ValueError('Unexpected held-out jump condition')
    train_actions = {tuple(row['actions']) for row in training}
    test_actions = {tuple(row['actions']) for row in test}
    return dict(command_overlap=0, canonical_coverage=len(canonical),
                shared_action_sequences=len(train_actions & test_actions),
                test_rows_with_action_sequence_seen_in_train=sum(tuple(row['actions']) in train_actions for row in test),
                explanation='Different commands may have identical action sequences. This is retained, not treated as a split failure.')


def prepare(raw, output, card):
    paths = {name: acquire(raw, name) for name in FILES}
    universe = parse(paths['tasks.txt'].read_bytes(), 'tasks.txt')
    canonical = {row['command']: row['actions'] for row in universe}
    if len(canonical) != len(universe) or len(universe) != 20910:
        raise ValueError('The canonical SCAN command universe changed')
    partitions = {}; audits = {}
    for split, files in SPLITS.items():
        train, test = (parse(paths[name].read_bytes(), name) for name in files)
        audits[split] = audit_pair(train, test, canonical, split)
        partitions[split] = {}
        for partition, rows in (('train', train), ('test', test)):
            path = output / split / (partition + '.jsonl'); path.parent.mkdir(parents=True, exist_ok=True)
            payload = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode('utf8')
            temporary = path.with_suffix('.jsonl.tmp'); temporary.write_bytes(payload); temporary.replace(path)
            partitions[split][partition] = dict(**statistics(rows), bytes=len(payload), sha256=sha256(path))
    result = dict(name='SCAN canonical simple, length and added-primitive-jump splits',
        revision=REVISION, repository='https://github.com/brendenlake/SCAN',
        paper='https://proceedings.mlr.press/v80/lake18a.html',
        source_files={name: dict(url=BASE+name, bytes=size, sha256=digest) for name,(size,digest) in FILES.items()},
        parser_sha256=sha256(Path(__file__)), canonical=statistics(universe), partitions=partitions, overlap_audit=audits,
        normalization='No source-line or row-order changes. JSON records retain every repeated row, source line, command and action list.',
        partition='Official train/test memberships and multiplicities. No custom validation split or neural model selection at acquisition.',
        license='Upstream BSD license for CommAI-env software, Facebook 2016-present; the exact notice is preserved in licenses/SCAN-BSD.txt.',
        license_metadata_note='GitHub reports NOASSERTION; this record preserves the source notice rather than inventing a publisher SPDX assertion.',
        limitations=['Artificial grammar-generated instruction/action pairs; not human speech or general conversation.',
            'SCAN has no physical world observations, learned gait or body dynamics.',
            'The oracle is a source-integrity check, not a learned-model result or a fallback decoder.',
            'Jump, run and look actions have no validated physical implementation in FLM; walk/turn counts indicate only possible later coverage.',
            'These three splits reuse the same command universe and are separate experiments, not independent corpora.',
            'This is benchmark preparation only. No FLM, GRU or transformer has been trained or evaluated on SCAN.'])
    write_json(card, result)
    compact = {split: {part: {key: row[key] for key in ('rows', 'unique_commands', 'repeated_rows', 'walk_turn_only')}
                       for part, row in parts.items()} for split, parts in partitions.items()}
    print(json.dumps(dict(canonical_commands=len(canonical), partitions=compact, overlap_audit=audits), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw/scan') / REVISION)
    parser.add_argument('--output', type=Path, default=Path('data/processed/scan'))
    parser.add_argument('--card', type=Path, default=Path('data/cards/scan.json'))
    args = parser.parse_args(); prepare(args.raw, args.output, args.card)


if __name__ == '__main__':
    main()
