from src.evaluator import Evaluator
from src.spellChecker import SpellChecker
from src.preSuggesters import LevenshteinPreSuggester, NGrammsPreSuggester

ev = Evaluator('test_data/batch0.tab.txt', verbose='all')
sc = SpellChecker(LevenshteinPreSuggester)
print(ev.evaluate(sc))

sc = SpellChecker(NGrammsPreSuggester)
print(ev.evaluate(sc))
