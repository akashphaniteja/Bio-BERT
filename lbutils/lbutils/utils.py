import re
import json
class DataCleaning:
    """
    Class to extract numbers for evaluation
    """
    def __init__(self, df, log = False):
        """ Extracts annotations into a dataframe from Labelbox NER json Format
    
        Parameters
        ----------
        df : pandas.core.frame.Dataframe
            Dataframe output from the process_lbexport function
        
        """
        self.df = df
        self.log = log
        self.req_cols = self.df.columns.difference(['PMID','abstract', 'num_arms_in_study','group1','group2'])
        self.units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.scales = ["hundred", "thousand", "million", "billion", "trillion"]
        self.ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}

        self.full_valid_vocab = set()
        self.full_valid_vocab.update(self.units + self.tens + self.scales + ['and','ieth','th','y'] + list(self.ordinal_words.keys()))
    
    def text2int(self, textnum, raw_text, convert = False):
        """ Converts text to integer 

        If able to convert:
            If convert=True
                returns number, True
            else
                returns textnum, True

        If unable to convert:
            returns raw_text, False
        
        Parameters
        ----------
        textnum : str
            cleaned string(words of interest) in lowercase to convert
        raw_text : str
            uncleaned(annotation) string
        convert : bool
            return converted string or not
        
        
        Returns
        -------
        text/number : str/int
            Returns text/number depending on `convert`, Refer Function Definition for details
        able_to_convert : bool
            Returns True/False depending on able to convert or not
        """     
        cleaned_text = textnum
        textnum = textnum.lower()

        numwords = {}
        numwords['and'] = (1, 0)
        for idx, word in enumerate(self.units): numwords[word] = (1, idx)
        for idx, word in enumerate(self.tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(self. scales): numwords[word] = (10 ** (idx * 3 or 2), 0)
        
        ordinal_endings = [('ieth', 'y'), ('th', '')]

        textnum = textnum.replace('-', ' ')
        textnum = textnum.replace('‐', ' ')

        current = result = 0
        for word in textnum.split():
            if word in self.ordinal_words:
                scale, increment = (1, self.ordinal_words[word])
            else:
                for ending, replacement in ordinal_endings:
                    if word.endswith(ending):
                        word = "%s%s" % (word[:-len(ending)], replacement)
                
                if word not in numwords:
                    return raw_text, False
                
                scale, increment = numwords[word]

            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        
        # Able to convert
        if convert:
            return result + current, True
        return cleaned_text, True


    def build_vocab(self):
        """
        Builds a Invalid Vocabulary from Feature

        Currently only works for the feature - total_sample_size
        """
        feature_name = 'total_sample_size'
        vocab = set()
        for index in self.df.index:
            feature = feature_name
            if not isinstance(self.df[feature][index], float):
                annot_list = json.loads(self.df[feature][index])
                abstract = self.df['abstract'][index]
                for annot in annot_list:
                    start, end = annot
                    s = abstract[start:end].split(' ')
                    for word in s:
                        vocab.add(word.replace(',', ''))
        
        # Keep only invalid vocab
        invalid_vocab = set()
        for i in vocab:
            _, valid = self.text2int(i.lower(), i)
            if not i.isnumeric() and not valid:
                # Edge case
                if '-' in i:
                    if any(self.text2int(s.lower(),'')[1] for s in i.split('-')):
                        continue
                invalid_vocab.add(i)
        
        # Remove number vocabulary or substring
        self.invalid_vocab = {i for i in invalid_vocab if all(i.lower() not in v.lower() for v in self.full_valid_vocab)}

    def clean_n(self, s, convert = False):
        """
        Cleans group size entity returns number

        Parameters 
        ----------
        s : str
            String to convert
        convert : bool
            Convert the word to number or not default - False
        
        Returns
        -------
        s : string
            converted number, if unable returns `raw_text`

        parsed : bool
            able to convert or not
        
        """
        raw_text = s
        # Remove symbols except -,=
        s = re.sub(r'[^,-=\w]', '', s)
        s = s.replace(' ','')
        s = s.replace('n=','')
        s = s.replace('patients','')
        s = s.replace('subjects','')
        s = s.replace('men','')
        s = s.replace('women','')
        if s.isnumeric():
            return s, True
        else:
            return self.text2int(s, raw_text, convert = convert)
    
    def clean_n_response(self, s, convert = False):
        """
        Cleans response size entity - returns number

        Parameters 
        ----------
        s : str
            String to convert
        convert : bool
            Convert the word to number or not default - False
        
        Returns
        -------
        s : string
            converted number, if unable returns `raw_text`

        parsed : bool
            able to convert or not
        
        """
        raw_text = s
        # Remove symbols except -,=
        s = s.replace(' ','')
        s = s.replace('patients','')
        s = s.replace('subjects','')
        s = s.replace('men','')
        s = s.replace('women','')
        if s.isnumeric():
            return s, True
        else:
            return self.text2int(s, raw_text)
    
    def clean_sample_size(self, s, convert = False):
        """
        Cleans sample_size entity returns number

        Parameters 
        ----------
        s : str
            String to convert
        convert : bool
            Convert the word to number or not default - False
        
        Returns
        -------
        s : string
            converted number, if unable returns `raw_text`

        parsed : bool
            able to convert or not
        
        """
        raw_text = s

        # Get tokens in annotation
        tokens = s.split(' ')

        # Replace perfect matches
        for token in tokens:
            if token in self.invalid_vocab:
                s = s.replace(token, '')
        s = s.strip()

        if s.isnumeric():
            return s, None
        else:
            return self.text2int(s, raw_text, convert = convert)

    def get_number_from_span(self, span):
        """
        Gets the first value encountered from span

        If more than 2 values are there, returns None

        Parameters 
        ----------
        span : str
            Span to extract from
        
        Returns
        -------
        s : string
            extracted value

        convert : bool
            able to extract or not
        
        """
        all_numbers = re.findall('\d*[.·]?\d+', span)
        if all_numbers and len(all_numbers) <= 2:
            return all_numbers[0], True
        else:
            return None, False

    def get_new_span(self, start, end, old_span, new_span, feature):
        """
        Returns the new indexes for `20` from `20 patients`

        Parameters 
        ----------
        start : int
            old start index
        end : int
            old start index
        old_span : str
            old span
        new_span : str
            new span
        feature : str
            feature name

        Returns
        -------
        start : int
            new start index
        end : int
            new end index
        
        """
        new_relative_index = re.search(new_span, old_span).span()
        new_index = new_relative_index[0] + start, new_relative_index[1] + start
        return new_index

    def extract_all(self):
        """
        Extracts the numbers from spans and fixes the indexes

        Returns
        -------
        df : pandas.core.frame.Dataframe
            cleaned dataframe for evaluation
        
        """
        for index in self.df.index:
            for feature in self.req_cols:
                if not isinstance(self.df[feature][index], float):
                    annot_list = json.loads(self.df[feature][index])
                    abstract = self.df['abstract'][index]

                    for i, annot in enumerate(annot_list):
                        start, end = annot
                        span = abstract[start:end]
                        parsed = False
                        if feature in ['g1_n', 'g2_n']:
                            new_span, parsed = self.clean_n(span)
                        elif feature in ['g1_n_response', 'g2_n_response']:
                            new_span, parsed = self.clean_n_response(span)
                        elif feature == 'total_sample_size':
                            new_span, parsed = self.clean_sample_size(span)
                        elif feature in self.df.columns.difference(['group1', 'group2','g1_n','g2_n','g1_n_response','g2_n_response', 'total_sample_size']):
                            new_span, parsed = self.get_number_from_span(span)
                        else:
                            raise Exception(f'Invalid Column Found! {feature}')
                        
                        # If able to parse
                        if parsed and len(span) - len(new_span):            
                            new_start, new_end = self.get_new_span(start, end, span, new_span, feature)
                            # See parsed old vs new
                            if self.log:
                                print([abstract[start:end], abstract[new_start:new_end]])
                            start, end = new_start, new_end

                        
                        annot_list[i] = [start, end]

                    self.df[feature][index] = json.dumps(annot_list)

        return self.df

                



