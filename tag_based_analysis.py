#Cooper Union Natural Language Processing Final Project
#Author: Eui Han
#Version: 04/26/16

from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import operator
import json
import random
import math

nltk_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
nltk_stemmer = nltk.stem.porter.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(['food', 'drink', 'place', 'one', 'us', 'would',
                  'service', 'restaurant', 'get', 'order', 'www',
                  'http', 'yelp', 'com', 'got'])
stopwords.remove('not')

class Reader(object):

    def read(text):
        text = text.lower()
        vocabs = nltk_tokenizer.tokenize(text)
        vocabs = [word for word in vocabs if word not in stopwords]
        stemmed = list()
        for v in vocabs:
            stemmed.append(nltk_stemmer.stem(v))
        #stemmed2 = nltk.ngrams(stemmed,2)
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

           # for word in bigrams:
           #     if word in mybag.keys():
           #         mybag[word] += 7 
           #     else:
           #         mybag[word] = 5

            for word in trigrams:
                if word in mybag.keys():
                    mybag[word] += 50 
                else:
                    mybag[word] = 1 
            
    def save(self):

        self.pos_bag = sorted(self.pos_bag.items(),
                         key=operator.itemgetter(1),
                         reverse=True)[0:1200]
        self.neg_bag = sorted(self.neg_bag.items(),
                         key=operator.itemgetter(1),
                         reverse=True)[0:1200]

        with open('./dicts/pos_feature.yml', 'w') as pos_feature:
            for tuples in self.pos_bag:
                outstring = '{}: {}\n'.format(tuples[0], tuples[1])
                pos_feature.write(outstring)

        with open('./dicts/neg_feature.yml', 'w') as neg_feature:
            for tuples in self.neg_bag:
                outstring = '{}: {}\n'.format(tuples[0], tuples[1])
                neg_feature.write(outstring)

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
        total_score = 0
        prev_word = None
        for current_word in text:
            current_score = 0
            #print('current word is {}'.format(current_word))
            if current_word in self.pos_features:
                current_score += math.log2(self.pos_features[current_word])
            #    #print('+1')

            if current_word in self.neg_features:
                current_score -= math.log2(self.neg_features[current_word])
            #    print('-1')

            if prev_word is not None:
            #    print('prev word is {}'.format(prev_word))
                if prev_word in self.inc_features:
                    current_score *= 1.5 
            #        print('*2')
                elif prev_word in self.dec_features:
                    current_score /= 1.5 
            #        print('/2')
                elif prev_word in self.inv_features:
                    current_score *= -1.0
            #        print('-')
            prev_word = current_word
            total_score += current_score

        return total_score

           

if __name__ == "__main__":

    trainer = Trainer()
    # postagger = POSTagger() 
    prompt = input("Enter the path for the train file\n")

    with open(prompt, 'r') as train_file:
        i=0
        for line in train_file:
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
            if entity['stars'] == 3: continue
            
            ngrams = Reader.read(entity['text'])

            score = classifier.evaluate(ngrams)
            if score > 1 : prediction = 'pos'
            elif score < 1 : prediction = 'neg'
            else:  
                r = random.random()
                if r>0.5:
                    prediction = 'pos'
                else:
                    prediction = 'neg'

            print(entity['review_id'] + " "  + str(prediction))

            outstring = {
                            'review_id': entity['review_id'],
                            'stars': prediction
                        }
                        
            json.dump(outstring, output_file)
            output_file.write('\n')

