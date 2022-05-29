import json
from rouge import Rouge

# https://pypi.org/project/rouge/

def score_summary(gold, pred):
    rouge = Rouge()
    return rouge.get_scores(pred, gold, avg=True)

