from spylls.hunspell import Dictionary
from typing import List
from pyphonetics import RefinedSoundex
from textdistance import needleman_wunsch, damerau_levenshtein
import numpy as np
import csv


class SpellChecker:
    def __init__(self, presuggester, max_suggestions=10, dict_name: str = 'en_US',
                 freqs_path: str = 'unigram_freq.csv'):
        self.max_suggestions = max_suggestions
        self.dict = Dictionary.from_files(dict_name)
        self.rs = RefinedSoundex()
        self.words_freqs = self.get_words_freqs(freqs_path)
        self.presuggester = presuggester(self.dict, 1000)

    def __call__(self, word):
        if self.lookup(word):
            print(f'{word} is okay')
        else:
            print(f'{word} is not okay. Maybe you mean one of these words?')
            suggestions = self.suggest(word, print_features=False)
            for suggestion in suggestions[:self.max_suggestions]:
                print(f'  -{suggestion}')

    def __calc_features(self, suggestion: str, misspelling: str) -> List[float]:
        res = []
        for dist_func in [damerau_levenshtein.normalized_distance, self.rs.distance,
                          needleman_wunsch.normalized_distance]:
            res.append(dist_func(suggestion, misspelling))

        res.append(1 - self.words_freqs.get(suggestion, 0))
        return res

    def lookup(self, word):
        return self.dict.lookup(word)

    def suggest(self, misspelling: str, print_features: bool = False) -> List[str]:
        # ngram_suggestions = self.dict.suggester.ngram_suggestions(misspelling, set())
        # suggestions = np.unique([el.lower() for el in ngram_suggestions])
        suggestions = self.presuggester.get_suggestions(misspelling)
        if not len(suggestions):
            return list()

        features = np.array([self.__calc_features(suggestion, misspelling) for suggestion in suggestions])
        indxs = self.__calc_ranks(features)
        if print_features:
            [print(sug, feat) for sug, feat in zip(suggestions[indxs], features[indxs])]
        return suggestions[indxs].tolist()

    @staticmethod
    def get_words_freqs(freqs_path: str):
        words_freqs = {}
        cnt = 0
        with open(freqs_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, (word, freq) in enumerate(reader):
                if not i:
                    continue
                cnt += int(freq)
                words_freqs[word] = int(freq)
        for key in words_freqs.keys():
            words_freqs[key] /= cnt

        return words_freqs

    @staticmethod
    def __calc_ranks(features: np.ndarray) -> np.ndarray:
        return np.argsort(np.mean(features, axis=1))
