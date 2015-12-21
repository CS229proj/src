import numpy as np
from itertools import izip, chain,islice
from passage.preprocessing import *

class CharTokenize(Tokenizer):
    num_features = 1
    def __init__(self, max_features=9997, min_df=10, lowercase=True, character=False, charn=1):
        super(CharTokenize, self).__init__(max_features, min_df, lowercase, character)
        self.charn = charn

    def ntuples(self, lst, n):
        iters = izip(*[chain(islice(lst,i,None)) for i in range(n)])
        return [''.join(i) for i in iters]

    def fit(self, texts):
        if self.lowercase:
            texts = [text.lower() for text in texts]
        if self.character:
            tokens = [self.ntuples(list(text.decode("utf-8")), self.charn) for text in texts]
#             print tokens
        else:
            tokens = [tokenize(text) for text in texts]
        self.encoder = token_encoder(tokens, max_features=self.max_features-3, min_df=self.min_df)
        self.encoder['PAD'] = 0
        self.encoder['END'] = 1
        self.encoder['UNK'] = 2
        self.decoder = dict(zip(self.encoder.values(), self.encoder.keys()))
        self.n_features = len(self.encoder)
        print('self.n_features ', self.n_features)
        self.num_features = self.n_features
        return self

    def transform(self, texts):
        if self.lowercase:
            texts = [text.lower() for text in texts]
        if self.character:
            texts = [self.ntuples(list(text.decode("utf-8")), self.charn) for text in texts]
        else:
            texts = [tokenize(text) for text in texts]
        tokens = [[self.encoder.get(token, 2) for token in text] for text in texts]
        return tokens