import pandas as pd
from danlp.models import load_spacy_chunking_model
from nltk import Tree
from nltk import word_tokenize, pos_tag
from nltk.chunk.regexp import RegexpParser

train = pd.read_csv('data/train.csv',
                    encoding='UTF8',
                    index_col=False)

# pre-processing of PoS-tags
def add_pause_dots(s):
    if s['RedPoS'] == '.' or s['RedPoS'] == '..' \
            or s['RedPoS'] == '...' or s['RedPoS'] == '....' \
            or s['RedPoS'] == '.....' or s['RedPoS'] == '......':
        return 'break'
    elif s['RedPoS']==s['ortografi'] and s['RedPoS'][0].isupper():
        return 'EGEN'
    else:
        return s['RedPoS']


def get_data(df, row_name):
    # Group after turns
    turns = df.groupby(by=['turnummer'])
    list_of_sentences = []
    for name, turn in turns:
        if row_name == 'RedPoS':
            s = [(str(row.Index), row.RedPoS) for row in turn.itertuples()]
        else:
            s = [(str(row.Index), row.ortografi) for row in turn.itertuples()]
        list_of_sentences.append(s)
    return list_of_sentences


def read_file(filename):
    file = pd.read_csv(filename)
    file = file[file['turnummer'].notna()].sort_values(by=['turnummer', 'xmin'])
    file['RedPoS'] = file.RedPoS.fillna(file['ortografi'])
    file['RedPoS'] = file.apply(lambda row: add_pause_dots(row), axis=1)
    return file, get_data(file, 'RedPoS'), get_data(file, 'ortografi')

def create_freq_dict(data):
    pos_tags = set(data['RedPoS'])
    pos_groups = data.groupby('RedPoS')

    freq_dict = dict()
    for pos in pos_tags:
        group = pos_groups.get_group(pos)
        counts = group['label'].value_counts()
        freq_dict[pos] = str(counts.index.values[0])
    return freq_dict

train['RedPoS'] = train.apply(lambda row: add_pause_dots(row), axis=1)
freq_dict = create_freq_dict(train)


file_test, test_data, test_orto = read_file('data/test_set.csv')
file_test2, test_data2, test_orto2 = read_file('data/hardtest.csv')

rules = """

PRON_1: {<PRON_UBST> <INTERJ|break>* <SKONJ> <INTERJ|break>* <PRON_UBST>}
PRON_1: {<PRON_UBST> <INTERJ|break>* <PRON_UBST>}
PRON_1: {<EGEN_GEN> <INTERJ|break>* <EGEN>}
ADJ_1: {<ADJ> <INTERJ|break>* <ADJ>+}
ADJ_1: {<ADJ>}
DET: {<NUM_ORD|NUM|PRON_POSS|EGEN_GEN|N_GEN>}
DET2: {<PRON_DEMO|PRON_PERS>}
DET3: {<PRON_UBST>}
NP: {<DET2|PRON_1|DET|DET3> <INTERJ|break>* <N> <INTERJ|break>* <ADJ_1>}
NP: {<DET2|PRON_1|DET|DET3> <INTERJ|break>* <ADJ_1|DET|DET3> <INTERJ|break>* <N>+}
NP: {<DET2|PRON_1|DET3|DET> <INTERJ|break>* <DET>* <INTERJ|break>* <ADJ_1|DET3>+ <N>*}
NP: {<DET2|PRON_1|DET|DET3> <INTERJ|break>* <N>+}
NP: {<ADJ_1|DET3> <INTERJ|break>* <N>}
NP: {<PRON_1>}
NP: {<DET2> <INTERJ|break>* <DET3>}
NP: {<PRON_INTER_REL|EGEN|N|DET2|DET3>}
"""

parser = RegexpParser(rules)

tokenized = word_tokenize('I am a bird')
tags = pos_tag(tokenized)


def parse_sentences(data):
    chunked_sentences = []
    for s in data:
        chunked = parser.parse(s)
        chunked_sentences.append(chunked)
    return chunked_sentences


def IOB(list):
    return ['B-' + item if index == 0 else 'I-'+ item for index, item in enumerate(list)]

def add_chunks(chunked, df):
    series = pd.Series()
    for sentences in chunked:
        for chunk in sentences:
            if type(chunk) == Tree:
                label = chunk.label()
                leaves = chunk.leaves()
                iob = IOB([label for i in range(len(leaves))])
                ser = {int(x[0]): iob[i] for i, x in enumerate(leaves)}
                ser = pd.Series(ser)
                series = series.append(ser)
    df['chunk'] = series
    return df

def add_freq_chunks(data, df, freq_dict):
    series = pd.Series()
    for sentence in data:
        ser = {int(index): freq_dict.get(token,"O") for index, token in sentence}
        ser = pd.Series(ser)
        series = series.append(ser)
    df['freq_chunk'] = series
    return df

# Load the chunker using the DaNLP wrapperi
chunker = load_spacy_chunking_model()
nlp = chunker.model
def add_spacy_chunks(data, df):
    series = pd.Series()
    for sentence in data:
        stop = ['uh','øh','mm', '.', '..', '...', '....', '.....', '......']
        index = [int(x[0]) for x in sentence if x[1] not in stop]
        text1 = [s[1] for s in sentence if s[1] not in stop]
        text = ' '.join(text1)
        np_chunks = chunker.predict(text)
        doc = nlp(text)
        for token, nc in zip(doc, np_chunks):
            try:
                ser = {index[token.i]: nc}
            except:
                print(token.i)
            series = series.append(pd.Series(ser))
        df['spacy_chunk'] = series
    return df


def parse_and_out(file, data, data_2, name):
    chunked = parse_sentences(data)
    file = add_chunks(chunked, file)
    #file = add_spacy_chunks(data_2, file)
    file.to_csv(name + '.csv', columns=['ortografi', 'RedPoS', 'xmin', 'turnummer', 'speaker', 'chunk', 'spacy_chunk'],
                index=False)

def parse_and_out_test(file, data, data_2, name, freq_dict):
    chunked = parse_sentences(data)
    file = add_chunks(chunked, file)
    file = add_spacy_chunks(data_2, file)
    file = add_freq_chunks(data, file, freq_dict)
    file.to_csv(name + '.csv', columns=['ortografi', 'PoS', 'RedPoS', 'xmin', 'xmax',
                                        'turnummer', 'speaker', 'gold','chunk', 'spacy_chunk', 'freq_chunk'],
                index=False)


parse_and_out_test(file_test2, test_data2, test_orto2, 'test_out2', freq_dict)
parse_and_out_test(file_test, test_data, test_orto, 'test_out1', freq_dict)