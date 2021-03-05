from typing import List, Tuple

import pandas as pd
import spacy

Annotations = List[Tuple[int, int, str]]


# creating a blank model, could have started from something pretrained as well (look for models)
def create_blank_nlp(train_data: pd.Series) -> spacy.Language:
    """Create a blank NLP model for training"""
    nlp = spacy.blank("en")
    # add more rules for tokenizing
    infixes = nlp.Defaults.infixes + [r"[-/=)(><]"]
    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer
    # add ner component
    nlp.add_pipe("ner", last=True)
    ner = nlp.get_pipe("ner")
    for tags in train_data:
        for tag in tags:
            ner.add_label(tag[-1])
    return nlp
