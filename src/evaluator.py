from src.spellChecker import SpellChecker
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np


class Evaluator:
    def __init__(self, test_data_path: str, k_tuple: Tuple[int] = (1, 3, 5), verbose='mute'):
        with open(test_data_path, 'r') as f:
            self.data = tuple([tuple(el.strip('\n').split('\t')) for el in f.readlines()])

        self.k_tuple = k_tuple
        self.log_file = open('log.txt', 'w') if verbose == 'all' else None

    def __calc_metrics(self, gt: str, suggestions: List[str]) -> Dict[str, int]:
        res = {}
        for k in self.k_tuple:
            res[f'top_{k}_accuracy'] = int(gt in suggestions[:k])

        res['mean_place_in_top'] = suggestions.index(gt) + 1 if gt in suggestions else None
        res['mrr'] = 1 / (suggestions.index(gt) + 1) if gt in suggestions else 0
        return res

    def evaluate(self, sc: SpellChecker) -> Dict[str, float]:
        res = {}
        place_in_top_cnt = 0
        for misspelling, gt in tqdm(self.data):
            misspelling, gt = misspelling.lower(), gt.lower()
            suggestions = sc.suggest(misspelling)
            metrics = self.__calc_metrics(gt, suggestions)

            if metrics['mean_place_in_top'] is not None:
                place_in_top_cnt += 1
            else:
                metrics['mean_place_in_top'] = 0

            if self.log_file is not None:
                self.log_file.write(f'{misspelling} {gt} {suggestions}\n')
            for metric_name, metric in metrics.items():
                res.setdefault(metric_name, 0)
                res[metric_name] += metric

        for k in self.k_tuple:
            res[f'top_{k}_accuracy'] /= len(self.data)

        res['mean_place_in_top'] = res['mean_place_in_top']/place_in_top_cnt if place_in_top_cnt != 0 else None
        res['mrr'] /= len(self.data)

        for key in res.keys():
            res[key] = np.round(res[key], 3)

        return res

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()
