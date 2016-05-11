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
stopwords.remove('not')

class Reader(object):

    def read(text):
        text = text.lower()
        vocabs = nltk_tokenizer.tokenize(text)
        vocabs = [word for word in vocabs if word not in stopwords]
        stemmed = list()
        for v in vocabs:
            stemmed.append(nltk_stemmer.stem(v))
        stemmed3 = nltk.ngrams(stemmed,3)
        
        return stemmed, stemmed3

class Trainer(object):

    def __init__(self):
        self.neg_bag = dict()
        self.pos_bag = dict()


    def learn(self, text, star):
        if (star == 3):     # we dont want to learn from a neutral review 
            return          # because it has mixed pos and neg features
        else:
            if (star < 3):
                mybag = self.neg_bag
            else:
                mybag = self.pos_bag

            unigrams, trigrams = Reader.read(text)
            for word in unigrams:
                if word in mybag.keys():
                    mybag[word] += 1
                else:
                    mybag[word] = 1

           #for word in trigrams:
           #    if word in mybag.keys():
           #        mybag[word] += 30
           #    else:
           #        mybag[word] = 30
           
    def save(self):
        self.pos_bag = sorted(self.pos_bag.items(),
                         key=operator.itemgetter(1),
                         reverse=True)[:100]
        self.neg_bag = sorted(self.neg_bag.items(),
                         key=operator.itemgetter(1),
                         reverse=True)[:100]

        with open('./dicts/pos_feature.yml', 'wb') as pos_feature:
            for tuples in self.pos_bag:
                outstring = '{}: [{}]\n'.format(tuples[0], 'positive')
                pos_feature.write(bytes(outstring, 'UTF-8'))

        with open('./dicts/neg_feature.yml', 'wb') as neg_feature:
            for tuples in self.neg_bag:
                outstring = '{}: [{}]\n'.format(tuples[0], 'negative')
                neg_feature.write(bytes(outstring, 'UTF-8'))

class Classifier(object):
   
    def __init__(self):
        self.pos_features = dict()
        self.neg_features = dict()
        self.inc_features = dict()
        self.dec_features = dict()
        self.inv_features = dict()
 
        with open('./dicts/pos_feature.yml', 'r') as positives,\
             open('./dicts/neg_feature.yml', 'r') as negatives,\
             open('./dicts/inc.yml', 'r') as inc,\
             open('./dicts/dec.yml', 'r') as dec,\
             open('./dicts/inv.yml', 'r') as inv:
             self.pos_features = yaml.load(positives)
             self.neg_features = yaml.load(negatives)
             self.inc_features = yaml.load(inc)
             self.dec_features = yaml.load(dec)
             self.inv_features = yaml.load(inv)

    def evaluate(self, ngrams):
        total_score = 0
        for ngram in ngrams:
            total_score += self.score(ngram)
        return total_score

    def score(self, text):    
        current_score = 0
        prev_word = None
        for current_word in text:
            if current_word in self.pos_features:
                current_score += 1

            if current_word in self.neg_features:
                current_score -= 1

            if prev_word is not None:
                if prev_word in self.inc_features:
                    current_score *= 2.0
                elif prev_word in self.dec_features:
                    current_score /= 2.0
                elif prev_word in self.inv_features:
                    current_score *= -1.0
            prev_word = current_word

        return current_score

           
#class Splitter(object):
#
#    def __init__(self):
#        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
#        self.nltk_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#
#    def split1(self, text):
#        """
#        input format: a paragraph of text
#        output format: a list of lists of words.
#            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
#        """
#        sentences = self.nltk_splitter.tokenize(text)
#        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
#        return tokenized_sentences
#
#class POSTagger(object):
#
#    def __init__(self):
#        pass
#        
#    def pos_tag(self, sentences):
#        """
#        input format: list of lists of words
#            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
#        output format: list of lists of tagged tokens. Each tagged tokens has a
#        form, a lemma, and a list of tags
#            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
#                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
#        """
#
#        pos = [nltk.pos_tag(sentence) for sentence in sentences]
#        #adapt format
#        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
#        return pos
#
#class DictionaryTagger(object):
#
#    def __init__(self, dictionary_paths):
#        files = [open(path, 'r') for path in dictionary_paths]
#        dictionaries = [yaml.load(dict_file) for dict_file in files]
#        map(lambda x: x.close(), files)
#        self.dictionary = {}
#        self.max_key_size = 0
#        for dict_ in dictionaries:
#            for key in dict_:
#                self.dictionary[key] = dict_[key]
#
#    def tag(self, postagged_sentences):
#        return [self.tag_sentence(sentence) for sentence in postagged_sentences]
#
#    def tag_sentence(self, sentence, tag_with_lemmas=False):
#        """
#        the result is only one tagging of all the possible ones.
#        The resulting tagging is determined by these two priority rules:
#            - longest matches have higher priority
#            - search is made from left to right
#        """
#        print(sentence)
#        tagged_sentence = []
#        N = len(sentence)
#        if self.max_key_size == 0:
#            self.max_key_size = N
#        i = 0
#        for i in range(N):
#            j = min(i + self.max_key_size, N) #avoid overflow
#            tagged = False
#            while (j > i):
#                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
#                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
#                if tag_with_lemmas:
#                    literal = expression_lemma
#                else:
#                    literal = expression_form
#                if literal in self.dictionary:
#                    #self.logger.debug("found: %s" % literal)
#                    is_single_token = j - i == 1
#                    original_position = i
#                    i = j
#                    taggings = [tag for tag in self.dictionary[literal]]
#                    tagged_expression = (expression_form, expression_lemma, taggings)
#                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
#                        original_token_tagging = sentence[original_position][2]
#                        tagged_expression[2].extend(original_token_tagging)
#                    tagged_sentence.append(tagged_expression)
#                    tagged = True
#                else:
#                    j = j - 1
#            if not tagged:
#                tagged_sentence.append(sentence[i])
#                i += 1
#        return tagged_sentence
#
#def value_of(sentiment):
#    if sentiment == 'positive': return 1
#    if sentiment == 'negative': return -1
#    return 0
#
#def sentence_score(sentence_tokens, previous_token, acum_score):    
#    if not sentence_tokens:
#        return acum_score
#    else:
#        current_token = sentence_tokens[0]
#        tags = current_token[2]
#        token_score = sum([value_of(tag) for tag in tags])
#        if previous_token is not None:
#            previous_tags = previous_token[2]
#            if 'inc' in previous_tags:
#                token_score *= 2.0
#            elif 'dec' in previous_tags:
#                token_score /= 2.0
#            elif 'inv' in previous_tags:
#                token_score *= -1.0
#        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)
#
#def sentiment_score(review):
#    return sum([sentence_score(sentence, None, 0.0) for sentence in review])


if __name__ == "__main__":

    trainer = Trainer()
    # splitter = Splitter()
    # postagger = POSTagger()

    prompt = input("Enter the path for the train file\n")

    with open(prompt, 'r') as train_file:
        i=0
        for line in train_file:
            if i>3000: break
            print('training {}th review'.format(i))
            entity = json.loads(line)
            trainer.learn(entity['text'], entity['stars'])
            i+=1

    trainer.save()

    classifier = Classifier()

    test_path = input('Enter the path for the test file\n')
    output_path = input('Enter the path for the output file\n')

    with open(test_path, 'r') as test_file, open(output_path, 'w') as output_file:
        for text in test_file:
            entity = json.loads(text)
            #if entity['stars'] == 3: continue
            
            ngrams = Reader.read(entity['text'])
            #unigrams = splitter.split1(text)
            #trigrams = splitter.split3(text)
            #tagged_sentences = unigrams.extend(trigrams)
            #pprint(splitted_sentences)

            #pos_tagged_sentences = postagger.pos_tag(unigrams)
            #pprint(pos_tagged_sentences)

            #pprint(dict_tagged_sentences)
            #print(dict_tagged_sentences)

            score = classifier.evaluate(ngrams)
            if score > 0 : prediction = 'pos'
            elif score < 0 : prediction = 'neg'
            else: prediction = 'neutral'
            print(entity['review_id'] + " "  + str(prediction))

            outstring = {
                            'review_id': entity['review_id'],
                            'stars': prediction
                        }
                        
            json.dump(outstring, output_file)
            output_file.write('\n')

