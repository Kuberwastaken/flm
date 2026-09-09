"""Pinned WikiText-2 raw acquisition with official splits and lossless article grouping."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import requests
import pyarrow.parquet as pq
from .provenance import sha256, write_json

REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
FILES = {
    'train': (6357543, 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7'),
    'validation': (657209, '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c'),
    'test': (732610, '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'),
}
ROWS = {'train': 36718, 'validation': 3760, 'test': 4358}
ARTICLES = {'train': 600, 'validation': 60, 'test': 60}


def acquire(raw, split):
    size, checksum = FILES[split]
    name = f'{split}-00000-of-00001.parquet'; path = raw / name
    url = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-2-raw-v1/{name}'
    if not path.exists() or path.stat().st_size != size or sha256(path) != checksum:
        raw.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix('.part')
        with requests.get(url, stream=True, timeout=90) as response:
            response.raise_for_status(); written = 0
            with temporary.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    written += len(chunk)
                    if written > size: raise ValueError('WikiText file exceeds pinned size')
                    handle.write(chunk)
        if written != size or sha256(temporary) != checksum: raise ValueError('WikiText source checksum mismatch')
        temporary.replace(path)
    return path, dict(url=url, bytes=size, sha256=checksum)


def article_title(text):
    stripped = text.strip()
    # Main article headers have one equals sign on either side. Subsections
    # such as '= = History = =' are retained inside the current article.
    match = re.fullmatch(r'=\s+([^=].*?)\s+=', stripped)
    return match.group(1).strip() if match and not match.group(1).rstrip().endswith('=') else None


def group_articles(rows, split):
    documents = []; parts = []; title = None; start = 0
    for index, row in enumerate(rows):
        found = article_title(row)
        # Equals-delimited sports-table definitions also resemble headings.
        # Actual article headings are isolated by empty source rows.
        isolated = (index == 0 or not rows[index - 1].strip()) and (index + 1 == len(rows) or not rows[index + 1].strip())
        if not isolated: found = None
        if found and title is not None:
            documents.append(dict(title=title, text=''.join(parts), start_row=start, end_row=index))
            parts = []; start = index
        if found: title = found
        # Preserve every original row string, including its existing newlines.
        # Empty rows contribute the empty string; no inferred separators added.
        parts.append(row)
    if title is None: raise ValueError('No article headers found')
    documents.append(dict(title=title, text=''.join(parts), start_row=start, end_row=len(rows)))
    for index, document in enumerate(documents):
        document.update(id=f'wikitext2-{split}-{index:04d}', source='WikiText-2 raw v1',
            article_key=hashlib.sha256(document['title'].casefold().encode()).hexdigest(),
            text_sha256=hashlib.sha256(document['text'].encode()).hexdigest())
    if ''.join(d['text'] for d in documents) != ''.join(rows): raise ValueError('Article grouping changed source text')
    return documents


def prepare(raw, output, card):
    statistics, source_files, partitions = {}, {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for split in FILES:
        path, source = acquire(raw, split); rows = pq.read_table(path, columns=['text'])['text'].to_pylist()
        if len(rows) != ROWS[split] or not all(isinstance(row, str) for row in rows): raise ValueError('Unexpected WikiText row schema')
        documents = group_articles(rows, split); partitions[split] = documents
        if len(documents) != ARTICLES[split] or len({d['article_key'] for d in documents}) != len(documents):
            raise ValueError('Article boundary audit failed')
        destination = output / f'{split}.jsonl'
        destination.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in documents), encoding='utf8')
        statistics[split] = dict(rows=len(rows), articles=len(documents),
            words=sum(len(d['text'].split()) for d in documents),
            utf8_bytes=sum(len(d['text'].encode()) for d in documents),
            characters=sum(len(d['text']) for d in documents), sha256=sha256(destination),
            concatenated_source_sha256=hashlib.sha256(''.join(rows).encode()).hexdigest())
        source_files[split] = source
    overlaps = {}
    for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        title_overlap = {d['article_key'] for d in partitions[first]} & {d['article_key'] for d in partitions[second]}
        exact_overlap = {d['text_sha256'] for d in partitions[first]} & {d['text_sha256'] for d in partitions[second]}
        overlaps[f'{first}/{second}'] = dict(article_title_overlap=len(title_overlap), exact_article_overlap=len(exact_overlap))
        if title_overlap or exact_overlap: raise ValueError(f'Unexpected article overlap: {first}/{second}')
    result = dict(name='WikiText-2 raw v1', dataset='Salesforce/wikitext', config='wikitext-2-raw-v1', revision=REVISION,
        source='https://huggingface.co/datasets/Salesforce/wikitext', paper='https://arxiv.org/abs/1609.07843',
        source_files=source_files, statistics=statistics, overlap_audit=overlaps,
        normalization='None. Concatenate original row strings, preserve case, punctuation, Unicode, markup and whitespace; group at main article headings isolated by empty source rows.',
        partition='Official train/validation/test. Article grouping changes reset boundaries only, not text or split membership.',
        license_metadata=['CC-BY-SA-3.0', 'GFDL'],
        license_note='Publisher metadata lists CC BY-SA 3.0/GFDL, while card prose links CC BY-SA 4.0. Preserve the discrepancy. Raw and normalized corpus text are not redistributed by this repository.',
        limitations=['Written encyclopedic English, not conversational instruction data',
            'Raw means no unknown-word replacement; this release still contains WikiText markup and tokenization artifacts',
            'Title and exact-article checks do not exhaustively exclude quoted passages or near duplicates',
            'Our byte/subword likelihoods and article-reset protocol are not interchangeable with published word-level WikiText perplexity'])
    write_json(card, result)
    print(json.dumps(dict(statistics=statistics, overlap_audit=overlaps), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw/wikitext2'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/wikitext2'))
    parser.add_argument('--card', type=Path, default=Path('data/cards/wikitext2.json'))
    args = parser.parse_args(); prepare(args.raw, args.output, args.card)


if __name__ == '__main__': main()
