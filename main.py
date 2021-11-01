from src.spellChecker import SpellChecker
from src.preSuggesters import BasePreSuggester, LevenshteinPreSuggester, NGrammsPreSuggester

sc = SpellChecker(NGrammsPreSuggester)
test_words = ['summer', 'sammer', 'sumer', 'summe', 'smmer', 'samer']
for tw in test_words:
    sc(tw)
    print()


# from src.preSuggesters import LevenshteinPreSuggester
# from spylls.hunspell import Dictionary
# LevenshteinPreSuggester(Dictionary.from_files('en_US'), 10, 6).get_suggestions('sammer')
