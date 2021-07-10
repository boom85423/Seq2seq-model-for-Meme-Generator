from tqdm import tqdm
import numpy as np
import re
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from nltk.stem import PorterStemmer
from textblob import TextBlob
import random
import gensim
from textaugment import Wordnet, Translate
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import itertools
import time


contractions_dict = { "ain’t": "are not", "’s":" is", "aren’t": "are not", "can’t": "can not", "can’t’ve": "can not have", "‘cause": "because", "could’ve": "could have", "couldn’t": "could not", "couldn’t’ve": "could not have", "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hadn’t’ve": "had not have", "hasn’t": "has not", "haven’t": "have not", "he’d": "he would", "he’d’ve": "he would have", "he’ll": "he will", "he’ll’ve": "he will have", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have", "I’m": "I am", "I’ve": "I have", "isn’t": "is not", "it’d": "it would", "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have", "let’s": "let us", "ma’am": "madam", "mayn’t": "may not", "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have", "must’ve": "must have", "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have", "o’clock": "of the clock", "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have", "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have", "that’d": "that would", "that’d’ve": "that would have", "there’d": "there would", "there’d’ve": "there would have", "they’d": "they would", "they’d’ve": "they would have","they’ll": "they will",
        "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have", "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are", "we’ve": "we have", "weren’t": "were not","what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are", "what’ve": "what have", "when’ve": "when have", "where’d": "where did", "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have", "who’ve": "who have", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have", "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all", "y’all’d": "you all would", "y’all’d’ve": "you all would have", "y’all’re": "you all are", "y’all’ve": "you all have", "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have", "you’re": "you are", "you’ve": "you have", " af ": " as fuck ", " afk ": " as fuck ", " u ": " you ", " ur ": " your ", " thx ": " thanks ", "thx ": "thanks ", " im ": " I am ", "dont": "do not", "notifs": "notifications", "you re": "you are", "it's": "it is"}
contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))


def expand_contraction(sentence, contractions_re=contractions_re, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, sentence)


def check_spelling(sentence, TextBlob=TextBlob):
    textBlb = TextBlob(sentence)
    return textBlb.correct()


stemmer = PorterStemmer()
def check_lemma(sentence, stemmer=stemmer):
    sentence = " ".join([stemmer.stem(word) for word in sentence.split()])
    return sentence


wnet = Wordnet()
def augment_input(sent, num_aug=5):
    sent = sent.strip()
    augmented = [sent]
    for i in range(num_aug):
        _sent = wnet.augment(sent)
        augmented.append(_sent)
    return list(set(augmented))


def augment_output(sent, num_aug=5):
    sent_augmented = []
    sent_splitted = sent.split('[SEP]')
    first_augmented_sent = augment_input(sent_splitted[0])
    first_augmented_sent = list(set(first_augmented_sent))
    for first_aug in first_augmented_sent:
        sent = sent_splitted
        sent[0] = first_aug
        sent = " [SEP] ".join(sent_splitted)
        sent_augmented.append(sent)
    return sent_augmented


def augment(meme_input, meme_output):
    meme_input_augmented = {}
    meme_output_augmented = {}
    for key in tqdm(meme_input):
        input_sent = meme_input[key]
        output_sent = meme_output[key]

        input_sents = augment_input(input_sent)
        output_sents = augment_output(output_sent)

        for idx, (input_sent, output_sent) in enumerate(list(itertools.product(input_sents, output_sents))):
            meme_input_augmented[key+str(idx)] = input_sent
            meme_output_augmented[key+str(idx)] = output_sent
    return meme_input_augmented, meme_output_augmented


if __name__ == '__main__':
    # template >> instance >> sent1;sent2
    raw_meme_sentence = np.load('data/raw_meme_sentence.npy', allow_pickle=True).tolist()

    meme_sentence = {}
    for template in tqdm(raw_meme_sentence, desc="normalize sentence"):
        num_boxes = int(template.replace('.jpg', '').split('-')[-1])
        cur_image = 0
        for meme in raw_meme_sentence[template]:
            sentence = raw_meme_sentence[template][meme].split(';')
            if len(sentence) == num_boxes:
                normal_sentence = ""
                for sent in sentence:
                    # text normalization
                    sent = expand_contraction(sent.lower())
                    sent = check_spelling(sent)
                    sent = check_lemma(sent)
    
                    if len(normal_sentence) > 0:
                        normal_sentence += " [SEP] "
                        normal_sentence += sent
                    else:
                        normal_sentence += sent
                
                words = [len(sent.split()) for sent in normal_sentence.split('[SEP]')]
                if 0 not in words:
                    meme_sentence['%s-%s' % (template, cur_image)] = normal_sentence
                    cur_image += 1


    # only including meme has more than two sentences
    meme_input = {}
    meme_output = {}
    for meme in tqdm(meme_sentence):
        sent = meme_sentence[meme]
        sent_splitted = sent.split('[SEP]')
        if len(sent_splitted) > 1:
            meme_input[meme] = sent_splitted[0]
            meme_output[meme] = " [SEP] ".join(sent_splitted[1:])
    
    
    # text augmentation
    meme_input, meme_output = augment(meme_input, meme_output)


    # calculate max_length of inputs and outputs for training
    length_input, length_output = {}, {}
    total_input, total_output = 0, 0
    for meme in tqdm(meme_input):
        sent_input = meme_input[meme]
        sent_output = meme_output[meme]
        sent_input_token = tokenizer(sent_input)['input_ids']
        sent_output_token = tokenizer(sent_output)['input_ids']
        length_input[meme] = len(sent_input_token)
        length_output[meme] = len(sent_output_token)
        total_input += len(sent_input_token)
        total_output += len(sent_output_token)
    final_length_input = round(total_input / len(length_input))
    final_length_output = round(total_output / len(length_output))
    print('final length of input:', final_length_input)
    print('final length of output:', final_length_output)
    
    
    # only include data which words length smaller than the mean
    filtered_meme_input = {}
    filtered_meme_output = {}
    for meme in tqdm(length_input):
        if (length_input[meme] <= final_length_input) and (length_output[meme] <= final_length_output):
            filtered_meme_input[meme] = meme_input[meme]
            filtered_meme_output[meme] = meme_output[meme]
    
    
    # data splitting
    keys = list(filtered_meme_input.keys())
    train_keys = set(np.random.choice(keys, int(len(keys)*0.8), replace=False))
    test_keys = set(keys) - train_keys
    
    meme_input_train, meme_output_train = {}, {}
    meme_input_test, meme_output_test = {}, {}
    for key in keys:
        if key in train_keys:
            meme_input_train[key] = filtered_meme_input[key]
            meme_output_train[key] = filtered_meme_output[key]
        else:
            meme_input_test[key] = filtered_meme_input[key]
            meme_output_test[key] = filtered_meme_output[key]
    
    np.save('data/meme_input_train.npy', meme_input_train)
    np.save('data/meme_input_test.npy', meme_input_test)
    np.save('data/meme_output_train.npy', meme_output_train)
    np.save('data/meme_output_test.npy', meme_output_test)



