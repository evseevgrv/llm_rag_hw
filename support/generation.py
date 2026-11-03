import torch
from transformers import LogitsProcessor


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        for i, row in enumerate(input_ids):
            generated = row[self.prompt_length:].unique()
            scores[i, generated] -= self.penalty
        return scores
