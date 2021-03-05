import json
import re
import os
import csv
import random
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from lbutils.utils import DataCleaning

dtypes = np.dtype([
          ('PMID', str),
          ('abstract', str),
          ('num_arms_in_study', str),
          ('group1', str),
          ('group2', str),
          ('total_sample_size', str),
          ('g1_n', str),
          ('g2_n', str),
          ('g1_response_rate', str),
          ('g1_n_response', str),
          ('g2_response_rate', str),
          ('g2_n_response', str),
          ('response_p', str),
          ('g1_median_OS', str),
          ('g1_median_PFS', str),
          ('g2_median_OS', str),
          ('g2_median_PFS', str),
          ('OS_HR', str),
          ('PFS_HR', str),
          ('OS_p', str),
          ('PFS_p', str),
          ('g1_survival_rate', str),
          ('g2_survival_rate', str),
          ('survival_p', str),
          ('Other_HR',str)
          ])

def process_lbexport(json_file, remove_2plus = False):
    """ Extracts annotations into a dataframe from Labelbox NER json Format
    
    Logs Processed, Duplicate & Missing Annotation numbers.
    Skips Annotations which don't have PMID in the text
    Convers annotation to python indexing - +1 for end index

    Parameters
    ----------
    json_file : .json file
        Exported Annotations .json file from Labelbox
    remove_2plus : bool
        Remove >2 arms default - False
    
    Returns
    -------
    df : pandas.core.frame.Dataframe
        Dataframe containing annotations per sample per feature dumped into json format
    missing_annots : list
        List of PMIDs which did not have any annotation in exported data
    """

    remove_feature_overlaps = False

    def find_index_ofdict(feature_spans, dict_to_find):
        """Returns index of dict from list"""
        return next((i for i, item in enumerate(feature_spans) if item == dict_to_find), None)

    def overlap(span_list):
        """Check if overlap is present in list of start, end indexes"""
        n = len(span_list)
        for i in range(1, n):
            if span_list[i-1][1] > span_list[i][0]:
                return True
        return False

    def drop_overlapping(span_list):
        """Drops invalid overlaps for a particular feature considering length"""
        filtered_list = []
        spans = []
        for span in span_list:
            span_dict = defaultdict()
            start, end = span[0], span[1]
            span_dict['start'] = start
            span_dict['end'] = end
            span_dict['length'] = abs(start - end)
            span_dict['valid'] = 1
            spans.append(span_dict)

        all_possible_overlaps = list(combinations(spans, 2))
        
        for span1, span2 in all_possible_overlaps:
            if span1['valid'] and span2['valid']:
                if (span2['start'] > span1['end']) or (span2['end'] < span1['start']):
                    # Both spans are valid
                    pass
                else:
                    # Prefer longer span
                    if span1['length'] > span2['length']:
                    # Mark 2nd as invalid
                        index = find_index_ofdict(spans, span2)
                        spans[index]['valid']  = 0
                    elif span1['length'] < span2['length']:
                    # Mark 1st as invalid
                        index = find_index_ofdict(spans, span1)
                        spans[index]['valid']  = 0
                    else:
                        raise Exception("Fail")

        for span in spans:
            # If valid(not overlapping)
            if span['valid']:
                filtered_list.append([span['start'], span['end']])

        return filtered_list

    df = pd.DataFrame(np.empty(0, dtype=dtypes))
    missing_annot = []
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
        for i, row in enumerate(data):
            sample = defaultdict(list)
            abstract = row['Labeled Data']
            sample['abstract'] = abstract
            pmid_begin = re.search(r"\d", abstract).start()
            pmid_end = abstract.find('T') - 1
            pmid = str(abstract[pmid_begin:pmid_end])
            pmid = pmid.replace(" ","")

            # skip abstracts which don't have pmids
            if not isinstance(pmid, str) or not pmid.isdigit():
                pmid = int(hashlib.sha256(abstract.encode('utf-8')).hexdigest(), 16) % 10**10
                print(f'No PMID found : index - {i}, Hashed to a 10 digit number - {pmid} ')

            sample['PMID'] = pmid

            flag_class = 0
            flag_label = 0

            # Check if class(num_arms) is labelled
            if 'classifications' in row['Label']:
                flag_class = 1
                for label in row['Label']['classifications']:
                    if label['title'] == 'num_arms_in_study':
                        num_arms_in_study = label['answer']['value']
                        break
                assert num_arms_in_study in ['1','2','>2'],f"Num_arms contains something else! - {num_arms_in_study}"
                sample['num_arms_in_study'] = num_arms_in_study
        
            # Check if abstract has any annotated labels
            if 'objects' in row['Label']:
                flag_label = 1
                for feature in row['Label']['objects']:
                    feature_name = feature['title']
                    # skip PMID since already captured
                    if feature_name == 'PMID':
                        continue

                    start = feature['data']['location']['start']
                    # + 1 for python indexing [a:b]
                    end = feature['data']['location']['end'] + 1
                    # There can be multiple labels for a single feature
                    sample[feature_name].append([start, end])
                
                # Remove conflicting single feature overlaps - keep longer ones & remove duplicates
                for feature, annot_list in sample.items():
                    if feature not in ['PMID', 'abstract','num_arms_in_study'] and len(annot_list) > 1:
                        # Remove Duplicate spans
                        new_list = list(set(map(tuple, annot_list)))
                        new_list.sort()
                        
                        if remove_feature_overlaps:
                            if overlap(new_list):
                                # If there is overlap
                                new_list = drop_overlapping(new_list)
                        
                        sample[feature] = new_list


            if flag_class or flag_label:
                for key in sample:
                    if not isinstance(sample[key], str):
                        sample[key] = json.dumps(sample[key])
                df = df.append(sample, ignore_index = True)
            else:
                # has PMID but no classification label or annotations
                missing_annot.append(sample['PMID'])   
    
    print(f'Exported abstracts length - {len(data)}')
    count = df.duplicated(subset = ['PMID']).sum()
    if count:
        print('Duplicate PMIDs Found!, Deleting..')
        print(f'Duplicate count - {count}')
        df = df.drop_duplicates(subset=['PMID'], ignore_index= True)    
    
    req_cols = df.columns.difference(['PMID','abstract', 'num_arms_in_study'])

    if remove_2plus:
        plus2 = df.loc[df['num_arms_in_study'].isin(['>2'])]
        if len(plus2):
            print(f'Found >2 arms dropping. Count - {len(plus2)}')
            df = df.loc[df['num_arms_in_study'].isin(['1', '2'])]

    # No annotations but has num_arms label tho
    no_annots = df[df[req_cols].isnull().all(axis=1)]
    if len(no_annots):
        df = df.dropna(how = 'all', subset = req_cols)
        missing_annot += list(no_annots.PMID.values)

    df.reset_index(inplace = True, drop=True)

    print(f'Processed abstracts - {len(df)}')
    # Abstract which were exported but did not have any annotations
    print(f'Missing annotation abstracts - {len(missing_annot)}')

    return df, missing_annot

def fix_annots(df, val = False, log = False):
    """
    Inplace Fixes annotations
    - Fixes annotations having whitespaces

    Parameters 
    ----------
    df : pandas.core.frame.Dataframe
        Dataframe output from the process_lbexport function
    val : bool
        Number Extraction during Evaluation default - False
    
    Returns
    -------
    df : pandas.core.frame.Dataframe
        Cleaned and processed dataframe
    
    """

    fix_decimals = False

    columns = df.columns
    for index in df.index:
        abstract = df['abstract'][index]
        pmid = df['PMID'][index]
        for col in columns:
            # Skip nan values and parse only required labels
            if not isinstance(df[col][index],float) and col not in ['PMID', 'abstract','num_arms_in_study']:
                annot_list = json.loads(df[col][index])
                # One feature can have multiple annotation
                for i, annot in enumerate(annot_list):
                    start, end = int(annot[0]), int(annot[1])
                    span = abstract[start:end]
                    span_check = span.strip()
                    # check if string contains whitespace at begin/end
                    if span != span_check:
                        # Correct indexes
                        # Forward
                        for c in span:
                            if c == ' ':
                                start += 1
                            else:
                                break
                        # Backward
                        for c in reversed(span):
                            if c == ' ':
                                end -= 1
                            else:
                                break
                                      
                    if fix_decimals:
                        # For such cases annotated -> 0139, token -> .0139
                        if start and abstract[start - 1] == '.':
                            # Possible candidate
                            if span.isnumeric():
                                # Its a number
                                check_span = abstract[start - 10:end]
                                check_span = check_span.replace(" ", "")
                                check_span = check_span.replace(span, "")
                                length = len(span)
                                # False positive -> .25 patients
                                if check_span[-1] == '.' and check_span[-2] == '=':
                                    # 0030 -> .0030
                                    start -= 1
                
                    annot_list[i] = [start, end]

                # Fixing new annotations can make new duplicates
                annot_list = list(set(map(tuple, annot_list)))
                df[col][index] = json.dumps(annot_list)
    
    if val:
        dc = DataCleaning(df, log = log)
        # For total sample size
        dc.build_vocab()
        df = dc.extract_all()

    return df


    

def to_jsonl(df, file_name):
    """
    Converts the dataframe to jsonl evaluation format

    Parameters
    ----------
    df : pandas.core.frame.Dataframe
        Processed exported dataframe using process_json()
    file_name : str
        json filename
    """
    data = []
    cols = list(df.columns)

    for index in df.index:
        row = dict()
        abstract = df['abstract'][index]
        pmid = df['PMID'][index]

        row['id'] = pmid
        row['text'] = abstract
        predictions = []
        for col in cols:
            # Skip nan values and parse only required labels
            if not isinstance(df[col][index],float) and col not in ['PMID', 'abstract','num_arms_in_study']:
                annot_list = json.loads(df[col][index])
                for i, annot in enumerate(annot_list):
                    start, end = annot
                    predictions.append({'start' : start, 'end' : end, 'entity' : col})
        row['predictions'] = predictions
        data.append(row) 
    
    with open(file_name + '.json', 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def df_to_raw(df, file_name, file_format):
    """Converts the cleaned dataframe back to labelbox raw format
    
    - Decrements end index to go to labelbox format

    Parameters
    ----------
    df : pandas.core.frame.Dataframe
        Processed & fixed exported dataframe
    file_name : str
        filename
    file_format : str
        csv/json export
    """


    rct_pre = []
    for index in df.index:
        sample = defaultdict()

        abstract = df['abstract'][index]
        pmid = df['PMID'][index]
        num_arms_in_study = df['num_arms_in_study'][index]

        sample['id'] = pmid
        sample['Labeled Data'] = abstract
        sample['Label'] = dict()
        sample['Label']['classifications'] = []
        sample['Label']['classifications'].append({'title':'num_arms_in_study', 'value':'num_arms_in_study', 'answer':{
                'title':num_arms_in_study, 'value':num_arms_in_study
        }})
        sample['Label']['objects'] = []

        for col in df.columns:
            if not isinstance(df[col][index],float) and col not in ['PMID', 'abstract','num_arms_in_study']:
                annot_list = json.loads(df[col][index])

                for annot in annot_list:
                    start, end = annot

                    sample['Label']['objects'].append({
                        'title':col, 'value':col, 'data' : {
                            'location' : {
                                'start' : start,
                                'end' : end - 1
                            }
                        }
                    })
        rct_pre.append(dict(sample))
        
    if file_format == 'json':
        with open(file_name + '.json', 'w', encoding='utf-8') as f:
            json.dump(rct_pre, f)
    else:
        with open(file_name + '.csv', mode='w', encoding='utf-8') as csv_file:
            fieldnames = ['id', 'Labeled Data', 'Label']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for row in rct_pre:
                writer.writerow({'id' : row['id'], 'Labeled Data': row['Labeled Data'], 'Label': json.dumps(row['Label'])})

def merge_json_files(input_dir, output_file_name):
    """Merges multiple json exports from Labelbox

    Parameters
    ----------
    input_dir : pandas.core.frame.Dataframe
        Directory containing files to merge
    output_file_name : str
        output filename

    """
    all_data = []
    all_files = os.listdir(input_dir)
    
    for _file in all_files:
        with open(input_dir + '/' + _file, encoding = 'utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
    
    with open(output_file_name + '.json','w', encoding='utf-8') as f:
        json.dump(all_data, f)
    
