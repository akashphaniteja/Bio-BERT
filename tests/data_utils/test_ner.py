import warnings

import numpy as np
from spacy.lang.en import English
from spacy.tokens import Span

from src.data_utils import ner


class TestTaggedCorpus:
    def test_prop_arr_char_spans(self) -> None:
        x = "This is a test text."
        tagged_corpus = ner.TaggedCorpus(x, [], tokenizer=None)
        assert (
            tagged_corpus.arr_char_spans
            == np.array([[0, 4], [5, 7], [8, 9], [10, 14], [15, 19], [19, 20]])
        ).all()

    def test_prop_aligned_annotations_dict(self) -> None:
        text = "Murilo is testing some components for Omdena and RebootRx."
        annotations = [(0, 6, "PERSON"), (37, 44, "ORG"), (49, 58, "ORG")]
        tagged_corpus = ner.TaggedCorpus(text, annotations)
        char_spans = {
            "ORG": [(38, 44), (49, 58)],
            "PERSON": [(0, 6)],
        }
        token_seq = {
            "ORG": ["O", "O", "O", "O", "O", "O", "ORG", "O", "ORG", "ORG"],
            "PERSON": ["PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
        }
        assert sorted(tagged_corpus.aligned_annotations_dict.keys()) == sorted(
            char_spans.keys()
        )
        assert all(
            [
                sorted(tagged_corpus.aligned_annotations_dict[k]["char_spans"])
                == sorted(char_spans[k])
                for k in tagged_corpus.aligned_annotations_dict.keys()
            ]
        )
        assert all(
            [
                sorted(tagged_corpus.aligned_annotations_dict[k]["token_seq"])
                == sorted(token_seq[k])
                for k in tagged_corpus.aligned_annotations_dict.keys()
            ]
        )

    def test_to_tokens(self) -> None:
        x = "This is a token-test text."
        tagged_corpus = ner.TaggedCorpus(x, [])
        res = tagged_corpus.to_tokens()
        assert res == [
            "This",
            "is",
            "a",
            "token",
            "-",
            "test",
            "text",
            ".",
        ]

    def test_token_char_spans(self) -> None:
        x = "This is a test text."
        tagged_corpus = ner.TaggedCorpus(x, [])
        assert sorted(tagged_corpus.to_token_char_spans()) == sorted(
            [(0, 4), (5, 7), (8, 9), (10, 14), (15, 19), (19, 20)]
        )

    def test_to_dict(self) -> None:
        text = "Murilo is testing some components for Omdena and RebootRx."
        annotations = [(0, 6, "PERSON"), (37, 44, "ORG"), (49, 58, "ORG")]
        tagged_corpus = ner.TaggedCorpus(text, annotations)
        res = tagged_corpus.to_dict()
        assert res == {
            "ORG": ["O", "O", "O", "O", "O", "O", "ORG", "O", "ORG", "ORG"],
            "PERSON": ["PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
        }

    def test_to_doc(self) -> None:
        text = "Murilo is testing some components for Omdena and RebootRx."
        annotations = [(0, 6, "PERSON"), (37, 44, "ORG"), (49, 58, "ORG")]
        tagged_corpus = ner.TaggedCorpus(text, annotations)
        res = tagged_corpus.to_doc()
        res_ents = [(token.text, token.ent_type_) for token in res]
        assert sorted(res_ents) == sorted(
            [
                ("Murilo", "PERSON"),
                ("is", ""),
                ("testing", ""),
                ("some", ""),
                ("components", ""),
                ("for", ""),
                ("Omdena", "ORG"),
                ("and", ""),
                ("RebootRx", "ORG"),
                (".", "ORG"),
            ]
        )

    def test_to_iob_list(self) -> None:
        text = "Murilo is from São Paulo, a city in Brazil."
        annotations = [(0, 6, "PERSON"), (15, 24, "PLACE"), (35, 43, "PLACE")]
        tagged_corpus = ner.TaggedCorpus(text, annotations)
        res = tagged_corpus.to_iob_list()
        assert res == [
            "B-PERSON",
            "O",
            "O",
            "B-PLACE",
            "I-PLACE",
            "O",
            "O",
            "O",
            "O",
            "B-PLACE",
            "I-PLACE",
        ]

    def test_to_multi_iob_list(self) -> None:
        text = "Murilo is from São Paulo, Brazil."
        annotations = [
            (0, 6, "PERSON"),
            (15, 24, "CITY"),
            (26, 32, "COUNTRY"),
            (15, 32, "PLACE"),
        ]
        tagged_corpus = ner.TaggedCorpus(text, annotations)
        res, _ = tagged_corpus.to_multi_iob_list()
        assert res == [
            ("O", "O", "B-PERSON", "O"),
            ("O", "O", "O", "O"),
            ("O", "O", "O", "O"),
            ("B-CITY", "O", "O", "B-PLACE"),
            ("I-CITY", "O", "O", "I-PLACE"),
            ("O", "O", "O", "I-PLACE"),
            ("O", "B-COUNTRY", "O", "I-PLACE"),
            ("O", "O", "O", "O"),
        ]


def test_get_entities() -> None:
    x = """{ "objects": [
        { "value": "g2_n_response", "data": { "location": { "start": 500, "end": 501 } } },
        { "value": "g2_n", "data": { "location": { "start": 506, "end": 507 } } }
        ] }"""
    res = ner.get_entities(x)
    assert res == [(500, 501, "g2_n_response"), (506, 507, "g2_n")]


def test_rm_groups() -> None:
    x = [
        (500, 501, "g1_n_response"),
        (506, 507, "g2_n"),
        (101, 202, "group1"),
        (123, 456, "no_grp_here"),
    ]
    res = ner.rm_groups(x)
    assert sorted(res) == sorted(
        [
            (500, 501, "n_response"),
            (506, 507, "n"),
            (101, 202, "group"),
            (123, 456, "no_grp_here"),
        ]
    )


def test_rm_colliding() -> None:
    x = [(2, 6, "TAG1"), (6, 8, "TAG2"), (9, 12, "TAG3"), (9, 10, "TAG4")]
    res = ner.rm_colliding(x)
    assert sorted(res) == sorted([(2, 6, "TAG1"), (6, 8, "TAG2"), (9, 12, "TAG3")])


def test_doc2ents() -> None:
    nlp = English()
    doc = nlp("Mr. Bean flew to New York on Saturday morning.")
    doc.set_ents([Span(doc, 0, 2, "PERSON"), Span(doc, 4, 6, "PLACE")])
    res = ner.doc2ents(doc)
    assert res == [
        "B-PERSON",
        "I-PERSON",
        "O",
        "O",
        "B-PLACE",
        "I-PLACE",
        "O",
        "O",
        "O",
        "O",
    ]


def test_generate_examples() -> None:
    nlp = English()
    texts = ["Mr. Bean flew to New York.", "Murilo is da bomb from Brazil."]
    ann_char = [
        [(0, 8, "PERSON"), (17, 25, "PLACE")],
        [(0, 6, "PERSON"), (23, 29, "PLACE")],
    ]
    ann_token = [
        [(0, 2, "PERSON"), (4, 6, "PLACE")],
        [(0, 1, "PERSON"), (5, 6, "PLACE")],
    ]
    res = ner.generate_examples(texts=texts, entity_offsets=ann_char, nlp=nlp)
    docs_ref = list(nlp.pipe(texts))
    docs_pred = list(nlp.pipe(texts))

    for doc, ann in zip(docs_ref, ann_token):
        doc.set_ents([Span(doc, start, end, tag) for start, end, tag in ann])

    assert [ner.doc2ents(doc) for doc in docs_pred] == [
        ner.doc2ents(r.predicted) for r in res
    ]
    assert [ner.doc2ents(doc) for doc in docs_ref] == [
        ner.doc2ents(r.reference) for r in res
    ]
