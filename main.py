from src.spellChecker import SpellChecker

sc = SpellChecker()
test_words = ['summer', 'sammer', 'sumer', 'summe', 'smmer', 'samer']
gt = 'summer'
for test_word in test_words:
    suggestions = sc.suggest(test_word, True)
    print(f'Suggests for "{test_word}": {sc.suggest(test_word, True)[:10]}. '
          f'{suggestions.index(gt) + 1 if gt in suggestions else None}')
