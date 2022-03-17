from typing import List
import numpy as np
import pdb


def char_accuracy(generations: List[str], answers: List[str]):
    corrects = .0
    total = .0
    for gen, ans in zip(generations, answers):
        gen = np.array(list(gen))
        ans = np.array(list(ans))

        if len(gen) < len(ans):
            diff = len(ans) - len(gen)
            gen = np.concatenate((np.zeros(diff, dtype=int), gen))
        elif len(ans) < len(gen):
            diff = len(gen) - len(ans)
            ans = np.concatenate((np.zeros(diff, dtype=int), ans))

        labels = np.array([g == a for g, a in zip(gen, ans)])
        corrects += labels.sum()
        total += len(answers)
    score = corrects/total
    score = np.around(score, decimals=3)
    return score

