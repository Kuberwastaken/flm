"""Independent Python reference IDs for the browser's lossless BPE implementation."""
import json
import argparse
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flm.tokenizer import Lexicon

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokenizer',default='data/tokenizers/wikitext2-4096/tokenizer.json')
parser.add_argument('--output',type=Path,default=Path('tests/fixtures/tokenizer-parity.json'))
args=parser.parse_args()
lexicon = Lexicon(args.tokenizer)
texts = ['', 'Hello, world!', "We're testing a fly's vocabulary.", "WE'RE TESTING contractions: I'll, won't, isn't.",
    '  leading   and trailing spaces  ', '\n\nA\tB\r\nC\vD\fE', 'café 日本語 🪰 हिन्दी',
    '<|bos|> [BOS] [EOS]', '\x00\x01\x1f\x7f', '0123456789' * 20, 'a' * 150, '\u00a0\u2009\u2028\u2029',
    'https://example.test/some/path?first=2&second=three', 'a: what should we make?\nb: something simple.']
rng = random.Random(42)
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '.,!?\n\t\r-_=+/" + 'éß日本語हिन्दी🪰🧠👩\u200d💻\u00a0\u2009'
texts += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 150))) for _ in range(100)]
path = args.output; path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(dict(tokenizer_sha256=lexicon.sha256, cases=[dict(text=text, tokens=lexicon.encode(text)) for text in texts]), ensure_ascii=False, indent=2), encoding='utf8')
print(f'Wrote {len(texts)} reference cases.')
