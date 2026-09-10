import tempfile
from pathlib import Path
import unittest
from flm.babylm_audit import connect, add_file, summarize, normalized_key
from flm.provenance import sha256


class BabyAuditTests(unittest.TestCase):
    def test_overlap_normalization_counts_repetitions_and_resume_does_not_double_count(self):
        sentence = 'The small bird returns to the quiet garden every morning.'
        self.assertIsNone(normalized_key('yes, thank you.'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); train = root / 'train'; test = root / 'test'
            train.write_text(sentence + '\n' + sentence + '\nhello\n', encoding='utf8')
            test.write_text(sentence.upper().replace(' ', '  ') + '\n', encoding='utf8')
            db = connect(root / 'audit.sqlite')
            try:
                add_file(db, train, 'train-10m', sha256(train))
                add_file(db, train, 'train-10m', sha256(train))
                add_file(db, test, 'test', sha256(test))
                result = next(x for x in summarize(db)['comparisons'] if x['first']=='train-10m' and x['second']=='test')
                self.assertEqual(result['shared_unique_normalized_lines'], 1)
                self.assertEqual(result['occurrences_in_first'], 2)
                self.assertEqual(result['occurrences_in_second'], 1)
                self.assertEqual(result['words_in_second_matching_first'], 10)
                with self.assertRaises(ValueError): add_file(db, train, 'train-10m', 'changed')
            finally: db.close()


if __name__ == '__main__': unittest.main()
