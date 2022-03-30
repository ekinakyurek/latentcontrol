from typing import List
import numpy as np
import pdb


def char_accuracy(generations: List[str], answers: List[str]):
    corrects = .0
    total = .0
    for gen, ans in zip(generations, answers):
        gen = gen.replace(" ", "")
        ans = ans.replace(" ", "")

        if len(gen) < len(ans):
            diff = len(ans) - len(gen)
            gen = "0"*diff + gen
        elif len(ans) < len(gen):
            diff = len(gen) - len(ans)
            ans = "0"*diff + ans

        labels = [g == a for g, a in zip(gen, ans)]
        corrects += np.sum(labels)
        total += len(labels)

    score = corrects/total
    score = np.around(score, decimals=3)
    return score


def exact_accuracy(generations: List[str], answers: List[str]):
    corrects = .0
    total = .0
    for gen, ans in zip(generations, answers):
        gen = gen.replace(" ", "")
        ans = ans.replace(" ", "")
        corrects += (gen == ans)
        total += 1
    score = corrects/total
    score = np.around(score, decimals=3)
    return score