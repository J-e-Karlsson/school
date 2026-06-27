#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@lingfil.uu.se>
"""

# import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter


class GPTWrapper:
    def __init__(self):
        """Wrapper for loading and using GPT-2.
        This removes the need for knowledge of the underlying modules. All 
        input and outputs use native python data types."""
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        # TODO Use cuda if present
        # TODO Load larger GPT as option
        # TODO Shared cache for futurum
        
    def text2ids(self, text, include_bos=False):
        """Tokenize a piece of text, returns a list of token ids."""
        assert type(text) is str
        token_ids = self.tokenizer.encode(text)
        if include_bos:
            token_ids = [self.tokenizer.bos_token_id] + token_ids
        return token_ids

    def ids2text(self, token_ids):
        """Forms a piece of text from token ids."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def ids2tokens(self, token_ids):
        """Takes a list of token ids and returns a list of tokens as strings."""
        if type(token_ids) is int:
            token_ids = [token_ids]
        assert type(token_ids) is list, "token_ids should be a list"
        assert type(token_ids[0]) is int
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def _predict(self, token_ids):
        assert type(token_ids) is list, "Token ids should be given in a list."
        if len(token_ids) > 0:
            token_ids = [int(n) for n in token_ids]
        # Add bos if the list is empty
        if len(token_ids)==0:
            token_ids = [self.tokenizer.bos_token_id] + token_ids
        # Add bos if not present
        if token_ids[0] != self.tokenizer.bos_token_id:
            token_ids = [self.tokenizer.bos_token_id] + token_ids
        with torch.inference_mode():
            outputs = self.model(torch.IntTensor(token_ids))
        # if outputs.logits
        token_logits = outputs.logits[:-1, :]
        next_token_logits = outputs.logits[-1, :]
        return token_logits, next_token_logits

    def predict_probs(self, token_ids):
        """Predicts a probability distribution for the given tokens.
        If missing, the bos token will be added before running the prediction."""
        token_logits, next_token_logits = self._predict(token_ids)
        token_probs = F.softmax(token_logits, dim=1).numpy()
        if token_ids[0] == self.tokenizer.bos_token_id:
            token_ids = token_ids[1:]
        return [token_probs[i, token_id] for i, token_id in enumerate(token_ids)]

    def predict_next(self, token_ids=[]):
        """Predicts a probability distribution for the next token. Can be 
        called without the token_ids argument to get a first token."""
        token_logits, next_token_logits = self._predict(token_ids)
        # temperature = 1.0
        # next_token_logits = next_token_logits / temperature
        next_token_probs = F.softmax(next_token_logits, dim=-1).numpy()
        ret = Counter()
        for index, p in enumerate(next_token_probs):
            ret[index] = p
        return ret

    # def sample_predict(self, token_ids):
    #     """Predicts a probability distribution for the next token"""
    #     pass

    # @property
    # def bos_token_id(self):
    #     return self.tokenizer.bos_token_id

    def get_vocabulary(self):
        """Returns a copy of the vocabulary.
        The vocabulary is given as a dictionary with item pairs as (token, index)."""
        return self.tokenizer.vocab.copy()

