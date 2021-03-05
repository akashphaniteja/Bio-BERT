import json
import numpy as np
import pandas as pd
from collections import defaultdict
from lbutils.data_utils import dtypes

def process_jsonl(json_file):
    """
    Converts model output to dataframe

    Parameters 
    ----------
    json_file : str
        model output json file
    
    Returns
    -------
    df : pandas.core.frame.Dataframe
        Output Dataframe
    
    """
    data = []
    with open(json_file, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(np.empty(0, dtype = dtypes))
    for i,row in enumerate(data):
        sample = defaultdict(list)
        sample['PMID'] = row['id']
        sample['abstract'] = row['text']

        for pred in row['predictions']:
            sample[pred['entity']].append([pred['start'], pred['end']])
        
        for key in sample:
            if not isinstance(sample[key], str):
                sample[key] = json.dumps(sample[key])
        df = df.append(sample, ignore_index = True)
    
    return df
