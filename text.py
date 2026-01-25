from typing import Any,Sequence,List,Optional,Callable
from tqdm.notebook import tqdm
import logging

def read_glove_embeddings(embeddings_dict:dict,fpath:Optional[str]="glove.6B.50d.txt")->None:
    import numpy as np
    with open(fpath, encoding="utf8") as f:
        for line in tqdm(f,miniters=100):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except (ValueError, TypeError):
                continue
            embeddings_dict[word] = coefs
    logging.debug('Total %s word vectors.' % len(embeddings_dict))