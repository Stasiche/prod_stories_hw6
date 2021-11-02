from src.evaluator import Evaluator
from src.spellChecker import SpellChecker, SpellCheckerPure
from src.preSuggesters import LevenshteinPreSuggester, NGrammsPreSuggester, HunspellPreSuggester
from spylls.hunspell import Dictionary
from datetime import datetime
from tabulate import tabulate

ev = Evaluator('test_data/batch0.tab.txt', verbose='mute')
dic = Dictionary.from_files('en_US')

levi_presug = LevenshteinPreSuggester(dic, max_suggestions=500)
ngramms_presug = NGrammsPreSuggester(dic, max_suggestions=500)
hunspell_presug = HunspellPreSuggester(dic, max_suggestions=500)

experiment = (
    ('Levi without features', SpellCheckerPure(levi_presug)),
    ('Levi with features', SpellChecker(levi_presug)),
    ('Ngramms without features', SpellCheckerPure(ngramms_presug)),
    ('Ngramms with features', SpellChecker(ngramms_presug)),
    ('Hunspell without features', SpellCheckerPure(hunspell_presug)),
    ('Hunspell with features', SpellChecker(ngramms_presug))
)

all_metrics = []
for _, sc in experiment:
    s = datetime.now()
    all_metrics.append(ev.evaluate(sc))
    all_metrics[-1]['time,s'] = (datetime.now() - s).seconds

data = []
for (name, _), metrics in zip(experiment, all_metrics):
    data.append([name])
    for metric in metrics.values():
        data[-1].append(metric)

head = [''] + [metric_name for metric_name in all_metrics[0].keys()]

print(tabulate(data, headers=head, tablefmt='grid'))
