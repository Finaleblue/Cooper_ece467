#-*- coding: utf-8 -*-
"""
basic_sentiment_analysis
~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the code and examples described in 
http://fjavieralba.com/basic-sentiment-analysis-with-python.html

"""

from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import operator
import json

nltk_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
nltk_stemmer = nltk.stem.porter.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(['food', 'drink', 'place', 'one', 'us', 'would',
                  'service', 'restaurant', 'get', 'order', 'www',
                  'http', 'yelp', 'com'])

class Trainer(object):
    def __init__(self):
        self.neg_bag = dict()
        self.pos_bag = dict()

    def read(self, text):
        text = text.lower()
        vocabs = nltk_tokenizer.tokenize(text)
        vocabs = [word for word in vocabs if word not in stopwords]
        stemmed = list()
        for v in vocabs:
            stemmed.append(nltk_stemmer.stem(v))
        stemmed3 = nltk.ngrams(stemmed,3)
        
        return stemmed, stemmed3

    def learn(self, text, star):
        if (star == 3):     # we dont want to learn from a neutral review 
            return          # because it has mixed pos and neg features
        else:
            if (star < 3):
                mybag = self.neg_bag
            else:
                mybag = self.pos_bag

            unigrams, trigrams = self.read(text)
            for word in unigrams:
                if word in mybag.keys():
                    mybag[word] += 1
                else:
                    mybag[word] = 1

            for word in trigrams:
                if word in mybag.keys():
                    mybag[word] += 30
                else:
                    mybag[word] = 30
            
    def save(self):
        self.pos_bag = sorted(self.pos_bag.items(),
                         key=operator.itemgetter(1),
                         reverse=True)[:100]
        self.neg_bag = sorted(self.neg_bag.items(),
                         key=operator.itemgetter(1),
                         reverse=True)[:100]

        with open('./dicts/pos_feature.json', 'wb') as pos_feature:
            json.dump(self.pos_bag, pos_feature)

        with open('./dicts/neg_feature.json', 'wb') as neg_feature:
            json.dump(self.neg_bag, neg_feature)

class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

class POSTagger(object):

    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tagged_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        for i in range(N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tagged_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tagged_sentence.append(sentence[i])
                i += 1
        return tagged_sentence

def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):    
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])


if __name__ == "__main__":

    trainer = Trainer()
    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml', 
                                    'dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'])

    prompt = raw_input("Enter the path for the train file\n")

    with open(prompt, 'r') as train_file:
        i=0
        for line in train_file:
            if i>3000: break
            print 'training {}th review'.format(i)
            entity = json.loads(line)
            trainer.learn(entity['text'], entity['stars'])
            i+=1

    test_path = raw_input('Enter the path for the test file\n')
    output_path = raw_input('Enter the path for the output file\n')

    with open(test_path, 'r') as test_file, open(output_path, 'wb') as output_file:
        for text in test_file:
            entity = json.loads(text)
            if entity['stars'] == 3: continue
            
            splitted_sentences = splitter.split(text)
            #pprint(splitted_sentences)

            pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
            #pprint(pos_tagged_sentences)

            dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
            #pprint(dict_tagged_sentences)

            print("analyzing sentiment...")
            score = sentiment_score(dict_tagged_sentences)
            print(score)


