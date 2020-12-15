from argparse import ArgumentParser
from nltk.corpus import wordnet
import os
import torch
from torch.autograd import Variable

from Process import create_fields
from Beam import beam_search
from transformer import SimpleTransformer


def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0


def translate(sentence, model, opt, SRC, TRG):
    model.eval()

    sentences = sentence.lower().split(".")

    for sentence in sentences:
        preprocess_sentence = SRC.preprocess(sentence + '.')

        indexed = []
        # get the index of the text in each sentence
        for token in preprocess_sentence:
            if SRC.vocab.stoi[token] != 0 or opt.floyd == True:
                indexed.append(SRC.vocab.stoi[token])
            else:
                indexed.append(get_synonym(token, SRC))

        sentence = Variable(torch.LongTensor([indexed])).to(opt.device)
        sentence = beam_search(sentence, model, SRC, TRG, opt)
        return sentence


def main():
    parser = ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-device', default='cpu')
    parser.add_argument('-max_strlen', default=80)
    parser.add_argument('-is_hierarchical', action='store_true')
    opt = parser.parse_args()

    dims = 512
    heads = 8
    N = 6

    SRC, TRG = create_fields(opt)
    model = SimpleTransformer(len(SRC.vocab), len(TRG.vocab), dims, N, heads, opt)
    model.load_state_dict(torch.load(os.path.join(opt.load_weights, 'model_weights.pkl')))

    while True:
        text = input('enter an english sentence to translate to french: ')
        translate(text, model, opt, SRC, TRG)


if __name__ == '__main__':
    main()
