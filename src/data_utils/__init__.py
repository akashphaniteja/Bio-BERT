import logging
import sys
from typing import Any, Iterable, List, Tuple

import pandas as pd

from src.data_utils.ner import get_entities, rm_groups

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

Annotations = List[Tuple[int, int, str]]


def batch_data(
    iterable: Iterable[Any], batch_size: int = 64, compounding: int = 1
) -> Iterable[List[Any]]:
    """Batch data - allow for increasing batch sizes"""
    assert (
        compounding >= 1
    ), "Compounding must be equal to or greater than 1 (always increasing)"
    b = batch_size
    iter_length = len(iterable)
    for ndx in range(0, iter_length, batch_size):
        batch = iterable[ndx : min(ndx + round(b), iter_length)]
        b *= compounding
        yield batch


def labelbox(
    df: pd.DataFrame,
    increment_end_span: bool = True,
    rm_groups: bool = False,
    dedup: bool = False,
) -> pd.DataFrame:
    """Process the CSV from labelbox and return relevant info.
    Dataframe must contain columns"""

    def _get_info(df: pd.DataFrame) -> pd.DataFrame:
        """Process data - get the annotations and the PMID"""
        _df = df.copy()
        _df["annotations"] = _df["Label"].apply(get_entities)
        _df["pmid"] = _df["Labeled Data"].str.extract(r"PMID: (\d+)  TITLE.")
        _df["text"] = _df["Labeled Data"]
        return _df[["pmid", "annotations", "text", "Label", "Labeled Data"]]

    def _clean(df: pd.DataFrame, dedup: bool) -> pd.DataFrame:
        """Drop empty annotations and duplicates"""
        _df = df.copy()
        is_invalid = _df["annotations"].apply(
            lambda x: len(x) < 1 if isinstance(x, list) else False
        )
        is_duplicated = _df["pmid"].duplicated()
        _df = _df.dropna(subset=["annotations"])
        _df = _df[_df.annotations.apply(len) > 0]

        if dedup:
            _df = _df.drop_duplicates(subset=["pmid"])

        if is_invalid.sum() > 0 or is_duplicated.sum() > 0:
            logger.info("Some examples were removed:")
            logger.info(f"Found {is_invalid.sum()} examples with no annotations.")
            logger.info(f"Found {is_duplicated.sum()} PMID duplicates.")
        return _df

    data = df[["Labeled Data", "Label"]].copy()
    data = data.reset_index(drop=True)

    logger.info("Parse annotations and extract features")
    data = _get_info(df=data)

    logger.info("Clean data")
    data = _clean(df=data, dedup=dedup)

    if rm_groups:
        logger.info("Remove group information")
        data["annotations"] = data["annotations"].apply(rm_groups)

    if increment_end_span:
        logger.info("Increment ending span by one (to account for labelbox tagging)")
        data["annotations"] = data["annotations"].apply(
            lambda l: [(start, stop + 1, entity) for start, stop, entity in l]
        )

    assert data.notna().values.all(), "There are Null values"
    if dedup:
        assert not data.pmid.duplicated().any(), "There are duplicates"

    data = data.reset_index(drop=True)
    return data
