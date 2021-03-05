import argparse
import re
from lbutils.model_utils import process_jsonl
from lbutils.data_utils import to_jsonl, fix_annots

def setup_parser():
    parser = argparse.ArgumentParser(description = 'Extract numerical entities')

    parser.add_argument('pred_file', type = str
                        , help = 'Prediction file in jsonl format')
    parser.add_argument('out_file', type = str
                        , help = 'Filepath to write output file')
    parser.add_argument('--verbose', '-v', action='store_true', help = 'Display edits')

    return parser


if __name__ == '__main__':

    parser = setup_parser()
    args = parser.parse_args()

    model_output_file = args.pred_file
    cleaned_output_file = re.sub(r'.json|.jsonl','', args.out_file)

    # Parse model output to dataframe
    df_pred = process_jsonl(model_output_file)

    verbose = args.verbose

    if verbose:
        print('Before ------- After')
    # Extract numbers
    df_pred = fix_annots(df_pred, val = True, log = args.verbose)

    to_jsonl(df_pred, cleaned_output_file)



    


