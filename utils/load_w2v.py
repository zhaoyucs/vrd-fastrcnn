import torch
from transformers import BertModel, BertTokenizer, BertConfig
import gensim


def load_from_bert(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path)
    input_ids = torch.tensor(tokenizer.encode("test")).unsqueeze(0)
    outputs = model(input_ids)
    sequence_output = outputs[0]
    print(sequence_output.shape)


def load_from_word2vec(filename):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(filename)
    return w2v


if __name__ == '__main__':
    load_from_bert("/Users/zhaoyu/NLP/embeddings/bert-base-cased")
