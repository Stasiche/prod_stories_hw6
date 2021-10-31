from spylls.hunspell import Dictionary
from typing import List
from pyphonetics import RefinedSoundex
import Levenshtein as lev
from textdistance import needleman_wunsch
import numpy as np
from scipy.stats import hmean
import csv


class SpellChecker:
    def __init__(self, dict_name: str = 'en_US', freqs_path: str = 'unigram_freq.csv'):
        self.dict = Dictionary.from_files(dict_name)
        self.rs = RefinedSoundex()
        self.words_freqs = self.get_words_freqs(freqs_path)

    def __calc_features(self, suggest: str, misspelling: str) -> List[float]:
        res = []
        for dist_func in [lev.distance, self.rs.distance, needleman_wunsch]:
            res.append(dist_func(suggest, misspelling) + 1)

        res.append(1 / self.words_freqs.get(suggest, 1e-10))
        return res

    def suggest(self, misspelling: str, print_features: bool = False) -> List[str]:
        ngram_suggestions = self.dict.suggester.ngram_suggestions(misspelling, set())
        suggestions = np.array([el for el in ngram_suggestions if self.__filter(el)])
        features = np.array([self.__calc_features(suggestion, misspelling) for suggestion in suggestions])
        indxs = self.__calc_ranks(features)
        if print_features:
            [print(sug, feat, hmean(feat)) for sug, feat in zip(ngram_suggestions, features)]
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
    def __filter(suggestion: str) -> bool:
        return suggestion[0].islower()

    @staticmethod
    def __calc_ranks(features: np.ndarray) -> np.ndarray:
        return np.argsort(hmean(features, axis=1))
