import unittest
from flm.wikitext import article_title, group_articles


class WikiTextTests(unittest.TestCase):
    def test_main_headings_are_distinct_from_subsections(self):
        self.assertEqual(article_title(' = A title = \n'), 'A title')
        self.assertIsNone(article_title(' = = History = = \n'))
        self.assertIsNone(article_title(' = = = Details = = = \n'))
        self.assertIsNone(article_title('ordinary text = not a heading'))

    def test_grouping_preserves_all_bytes_and_empty_rows(self):
        rows = ['', ' = Café = \n', '', ' A paragraph.\n', ' = = Details = = \n', ' More.\n', '', ' = Next = \n', '', ' End.\n']
        documents = group_articles(rows, 'train')
        self.assertEqual(len(documents), 2)
        self.assertEqual(''.join(x['text'] for x in documents), ''.join(rows))
        self.assertEqual(documents[0]['start_row'], 0)
        self.assertEqual(documents[0]['end_row'], 7)
        self.assertEqual(documents[1]['start_row'], 7)

    def test_sports_glossary_is_not_an_article_boundary(self):
        rows = ['', ' = Hockey = \n', '', ' Note : Pos\n', ' = Position ; GP = \n', ' Games played\n']
        self.assertEqual(len(group_articles(rows, 'train')), 1)


if __name__ == '__main__': unittest.main()
