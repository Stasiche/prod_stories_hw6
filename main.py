from spylls.hunspell import Dictionary
from src.spellChecker import SpellChecker
from src.preSuggesters import NGrammsPreSuggester

dic = Dictionary.from_files('en_US')
ngramms_presug = NGrammsPreSuggester(dic, max_suggestions=500)

sc = SpellChecker(ngramms_presug)
test_words = ['summer', 'sammer', 'sumer', 'summe', 'smmer', 'samer']
for tw in test_words:
    sc(tw)
    print()
