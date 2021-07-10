from __future__ import absolute_import, division, print_function

import re
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_vocab_dict_from_file(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]:n for n in range(len(words))}
    return vocab_dict

UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def sentence2vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if words[-1] == '.':
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words]
    return vocab_indices

PAD_IDENTIFIER = '<pad>'
EOS_IDENTIFIER = '<eos>'

# def sentence2pos(sent):
#     sent_tokens = nltk.word_tokenize(sent)
#     pos_tag_list = nltk.pos_tag(sent_tokens)
#     sent_pos = []
#     for pos_tag in pos_tag_list:
#         pos = pos_tag[1]
#         if pos[0] in ['N', 'J']: #noun and adj
#             sent_pos.append(1)
#         elif pos[0] in ['V', 'I']



def preprocess_sentence(sentence, vocab_dict, T):
    # word_type = sentence2pos(sentence)
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    # # Append '<eos>' symbol to the end
    # vocab_indices.append(vocab_dict[EOS_IDENTIFIER])
    # Truncate long sentences
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pad>'
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices

def preprocess_sentence_lstm(sentence, vocab_dict, T):
    # word_type = sentence2pos(sentence)
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    # # Append '<eos>' symbol to the end
    # vocab_indices.append(vocab_dict[EOS_IDENTIFIER])
    # Truncate long sentences
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pad>'
    sequence_length = len(vocab_indices)
    if len(vocab_indices) < T:
        vocab_indices = vocab_indices + [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices))
    return vocab_indices, sequence_length