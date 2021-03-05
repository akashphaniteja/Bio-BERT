import json
import warnings
from itertools import combinations, groupby
from typing import Dict, Iterable, List, Tuple

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

Annotations = List[Tuple[int, int, str]]
AnnotationsDict = Dict[str, Iterable[int]]


class TaggedCorpus:
    def __init__(
        self,
        text: str,
        annotations: Annotations,
        tokenizer: spacy.tokenizer.Tokenizer = None,
    ) -> None:
        self.text = text
        self.annotations = annotations
        if tokenizer is None:
            nlp = spacy.blank("en")
            # add more rules for tokenizing
            infixes = nlp.Defaults.infixes + [r"[-/=)(><]"]
            infix_regex = spacy.util.compile_infix_regex(infixes)
            nlp.tokenizer.infix_finditer = infix_regex.finditer
            self.tokenizer = nlp.tokenizer
        else:
            self.tokenizer = tokenizer
        # additional properties
        self.arr_char_spans, self.tokens = self._get_char_spans_and_tokens(
            text=text, tokenizer=self.tokenizer
        )
        self.aligned_annotations_dict = self._align_span_dict(
            annotations=annotations, text=text, tokenizer=self.tokenizer
        )

    @staticmethod
    def _annotations2dict(annotations: Annotations) -> AnnotationsDict:
        """
        Go from annotations list to a dictionary
        :param annotations: Entity annotations [(start, stop, entity), ...]
        :return: Dictionary of annotations {entity: [(start, stop), ...]}
        """
        s = sorted(annotations, key=lambda a: a[-1])
        grp = groupby(s, key=lambda a: a[-1])
        return {key: [val[:-1] for val in vals] for key, vals in grp}

    @staticmethod
    def _dict2annotations(annotations_dict: AnnotationsDict) -> Annotations:
        """
        Go from annotations dictionary to a list
        :param annotations_dict: Dictionary of annotations {entity: [(start, stop), ...]}
        :return: Dictionary of annotations {entity: [(start, stop), ...]}
        """
        return [
            (span[0], span[1], tag)
            for tag, spans in annotations_dict.items()
            for span in spans
        ]

    def _get_char_spans_and_tokens(
        self, text: str, tokenizer: spacy.tokenizer.Tokenizer
    ) -> (np.ndarray, List[str]):
        """
        Get the character spans and the text for each token in the input text.
        :param text: Input text
        :return: 2 components - numpy 2D array - rows are tokens, columns are start, stop character span; list of tokens
        """
        doc = tokenizer(text)
        arr = doc.to_array(["IDX", "LENGTH"])
        tokens = [token.text for token in doc]
        return np.array([arr[:, 0], arr[:, 0] + arr[:, -1]]).T, tokens

    def _align_span_dict(
        self, annotations: Annotations, text: str, tokenizer: spacy.tokenizer.Tokenizer
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the aligned character spans - spans start at first character index and end on last characted index + 1.
        For example, in "test-token.", `token` spans from (5,10) - as index starts at zero.
        :param annotations: Annotation list
        :param text: Input text
        :return: Dictionary of dictionaries - for each entity get the list of character spans and token sequences
        (example: {entity: {char_spans: [(0, 6), ...], token_seq: [ENT_1, ...]})
        """

        def _consecutive(arr: np.ndarray) -> List[Tuple[int, ...]]:
            """Get tuples of the consecutive integers in 1D array"""
            idx_splits = (np.diff(arr) != 1).nonzero()[0] + 1
            arr_consec = np.split(ary=arr, indices_or_sections=idx_splits)
            return [tuple(a) for a in arr_consec]

        def _stich_spans(
            arr_spans: np.ndarray, tk_spans: np.ndarray
        ) -> List[Tuple[int, int]]:
            """Stich back together the spans of consecutive tokens."""
            first_and_last_idx = [
                (grp[0], grp[-1]) for grp in _consecutive(arr_spans.nonzero()[0])
            ]
            return [
                (token_spans[start, 0], token_spans[end, -1])
                for start, end in first_and_last_idx
            ]

        d_arr = {}
        annotations_dict = self._annotations2dict(annotations=annotations)
        token_spans, _ = self._get_char_spans_and_tokens(text=text, tokenizer=tokenizer)
        for tag, spans in annotations_dict.items():
            # span of entity must include start and end characters of token to be tagged
            is_span = tuple(
                ((token_spans[:, 0] >= span[0]) & (token_spans[:, 1] <= span[1]))
                for span in spans
            )
            # gather all the tokens that meet criteria
            is_tag = np.any(is_span, axis=0)
            arr_tagged_tokens = np.where(is_tag, tag, "O")
            # for subsequent tokens with the same entity, we want to join the spans
            aligned_char_span = (
                _stich_spans(arr_spans=is_tag, tk_spans=token_spans)
                if any(is_tag)
                else []
            )
            d_arr[tag] = {
                "char_spans": aligned_char_span,
                "token_seq": arr_tagged_tokens.tolist(),
            }
        return d_arr

    def to_tokens(self) -> List[str]:
        """Get the token list."""
        return self.tokens

    def to_token_char_spans(self) -> List[Tuple[int, int]]:
        """Get the character spans for each token."""
        return [(e[0], e[1]) for e in self.arr_char_spans.tolist()]

    def to_dict(
        self,
    ) -> Dict[str, List[str]]:
        """
        Get the entities in a dictionary.
        :return: Dictionary with token sequences (ex: {entity_1: [ENT1, ...], entity_2: [ENT2, ...], ...})
        """
        d = {k: v["token_seq"] for k, v in self.aligned_annotations_dict.items()}
        return d

    def to_doc(self, entity: str = None) -> spacy.tokens.Doc:
        """
        Get the entities in an annotated spaCy document.
        :param entity: Optional parameter - if set, return document annotated with only selected entity
        :return: SpaCy document annotated with entities
        """
        annotations_dict = {
            k: [(e[0], e[1]) for e in v["char_spans"]]
            for k, v in self.aligned_annotations_dict.items()
        }
        # get the character span entity annotations
        if entity is None:
            aligned_annotations = self._dict2annotations(
                annotations_dict=annotations_dict
            )
            annotations = rm_colliding(aligned_annotations)
            if sorted(annotations) != sorted(aligned_annotations):
                warnings.warn("Found overlapping spans. Keep only longest span.")
        else:
            annotations = [
                (span[0], span[1], entity) for span in annotations_dict[entity]
            ]

        # generate a blank spacy document and spans for the entity annotations
        doc = self.tokenizer(self.text)
        entity_spans = [
            doc.char_span(start, end, label=entity)
            for start, end, entity in annotations
        ]
        assert (
            None not in entity_spans
        ), f"Token with annotation {annotations[entity_spans.index(None)]} don't match token spans."
        # annotate document with entity spans
        doc.set_ents(entity_spans)
        return doc

    def to_iob_list(self, entity: str = None) -> List[str]:
        """
        Get the entities in a list of entities (IOB format).
        :param entity: Optional parameter - if set, return document annotated with only selected entity
        :return: List of strings where each string represents the entity associated with the token
        """
        doc = self.to_doc(entity=entity)
        return doc2ents(doc)

    def to_multi_iob_list(self) -> (List[Tuple[str, ...]], List[str]):
        """
        Get all the entities in a list of entities (IOB format).
        :return: :return: List of tuples where each tuple contains one element for each entity present in the document
        """
        all_entities = [
            self.to_iob_list(entity=entity)
            for entity in self.aligned_annotations_dict.keys()
        ]
        return list(zip(*all_entities)), list(self.aligned_annotations_dict.keys())


def get_entities(s: str) -> Annotations:
    """Process annotations to get in the format of (start, stop, label)"""
    d = json.loads(s)

    # check if dictionary is not empty
    if not d:
        return None

    assert isinstance(d["objects"], list), "Couldn't find `objects` list."
    return [
        (el["data"]["location"]["start"], el["data"]["location"]["end"], el["value"])
        for el in d["objects"]
    ]


def rm_groups(a: Annotations) -> Annotations:
    """Remove the group number info - ie: `g2_response_rate` becomes `response_rate`"""

    def _rm_grp_info(s: str) -> str:
        if s.startswith("g1_") or s.startswith("g2_"):
            return s[3:]
        elif s in ["group1", "group2"]:
            return "group"
        else:
            return s

    return list({(start, stop, _rm_grp_info(ent)) for start, stop, ent in a})


def rm_colliding(x: Annotations) -> Annotations:
    """Remove any duplicated annotations (one token cannot have two entities)"""

    def _is_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def _span_to_drop(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        return a if (a[1] - a[0]) < (b[1] - b[0]) else b

    def _rm_overlap(ranges: Iterable[Tuple[int, int]]):
        not_valid = [
            _span_to_drop(a, b) for a, b in combinations(ranges, 2) if _is_overlap(a, b)
        ]
        valid = [el for el in ranges if el not in not_valid]

        # check for other overlaps (in case of 3-way collisions)
        if not_valid:
            return _rm_overlap(valid)
        else:
            return valid

    # if duplicated spans then take first
    d = {(start, stop): ent for start, stop, ent in x[::-1]}
    valid_spans = _rm_overlap(d.keys())
    return [
        (start, stop, ent)
        for (start, stop), ent in d.items()
        if (start, stop) in valid_spans
    ]


def doc2ents(doc: Doc) -> List[str]:
    """Take a spaCy `Doc` and return a list of IOB token-based entities"""
    return [
        f"{token.ent_iob_}-{token.ent_type_}"
        if token.ent_type_
        else token.ent_iob_
        if token.ent_iob_
        else "O"
        for token in doc
    ]


def generate_examples(
    texts: Iterable[str], entity_offsets: Iterable[Annotations], nlp: Language
) -> List[Example]:
    if len(texts) != len(entity_offsets):
        raise ValueError("Lengths don't match.")
    # get docs (with predictions)
    docs = [nlp.make_doc(text) for text in texts]
    examples = [
        Example.from_dict(doc, {"entities": entity_offsets})
        for doc, entity_offsets in zip(docs, entity_offsets)
    ]
    return examples
