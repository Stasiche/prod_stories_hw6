import Levenshtein as lev
from heapq import heappush, heappushpop
import numpy as np


class BasePreSuggester:
    def __init__(self, dictionary, max_suggestions):
        self.dict = dictionary
        self.max_suggestions = max_suggestions

    def get_suggestions(self, misspelling: str) -> np.ndarray:
        raise NotImplementedError


class LevenshteinPreSuggester(BasePreSuggester):
    def __init__(self, dictionary, max_suggestions, distance_treshold: int = 6):
        super().__init__(dictionary, max_suggestions)
        self.th = distance_treshold

    def get_suggestions(self, misspelling: str) -> np.ndarray:
        suggestions = []
        for word in self.dict.dic.words:
            words_variants = [word.stem] + word.alt_spellings
            for w in words_variants:
                if word.stem.isnumeric() or len(w) < 3:
                    continue
                dist = -lev.distance(misspelling, w)
                if dist <= self.th:
                    if len(suggestions) < self.max_suggestions:
                        heappush(suggestions, (dist, w))
                    else:
                        heappushpop(suggestions, (dist, w))

        return np.unique([el[1].lower() for el in suggestions])


class NGrammsPreSuggester(BasePreSuggester):
    def get_suggestions(self, misspelling: str) -> np.ndarray:
        ngram_suggestions = self.dict.suggester.ngram_suggestions(misspelling, set())
        return np.unique([el.lower() for el in ngram_suggestions])[:self.max_suggestions]


class HunspellPreSuggester(BasePreSuggester):
    def get_suggestions(self, misspelling: str) -> np.ndarray:
        ngram_suggestions = self.dict.suggest(misspelling)
        return np.unique([el.lower() for el in ngram_suggestions])[:self.max_suggestions]
