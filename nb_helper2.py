from __future__ import division
import re, os, io, math
import operator
import json

from nltk.corpus import stopwords 
from nltk import tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer


class BagOfWords(object):
    def __init__(self):
        self.number_of_vocabs = 0
        self.bag_of_words = {}
        
    def __add__(self, other):
        erg = BagOfWords()
        sum = erg.bag_of_words
        for key in self.bag_of_words:
            sum[key] = self.bag_of_words[key]
            if key in other.bag_of_words:
                sum[key] += other.bag_of_words[key]
        for key in other.bag_of_words:
            if key not in sum:
                sum[key] = other.bag_of_words[key]
        return erg
        
    def put_in(self,word,ngram):
        self.number_of_vocabs += 1
        if word in self.bag_of_words:
            self.bag_of_words[word] += ngram
        else:
            self.bag_of_words[word] = ngram
    
    def len(self):
        return len(self.bag_of_words)
    
    def words(self):
        return self.bag_of_words.keys()
    
        
    def get_bag(self):
        return self.bag_of_words
        
    def term_freq(self, word):
        if word in self.bag_of_words:
            return self.bag_of_words[word]
        else:
            return 0


class Document(object):
    vocabs = BagOfWords()
    term_weight ={} 
    def __init__(self, vocabulary):
        self.name = ""
        self.document_class = None
        self.word_bag = BagOfWords()
        self.number_of_docs = 1
        Document.vocabs = vocabulary
        Document.stops = set(stopwords.words('english'))
        Document.stops.update(['food', 'drink', 'place', 'one', 'us', 'would',
                               'service', 'restaurant', 'get', 'order', 'www',
                               'http', 'yelp', 'com'])

    def read(self, text, learn=False):
        text = text.lower()
        st = PorterStemmer()
        tknzr = tokenize.RegexpTokenizer(r'\w+')
        vocabs = tknzr.tokenize(text)
        vocabs = [word for word in vocabs if word not in Document.stops]
        stemmed = []
        for v in vocabs:
            stemmed.append(st.stem(v))
        stemmed3 = ngrams(stemmed, 3)
        self.number_of_vocabs = 0
        #vocabs = list(ngrams(vocabs,2))
        #print(vocabs[2])
        for word in stemmed:
            if learn:
                Document.vocabs.put_in(word,1)
                
            self.word_bag.put_in(word,1)
            self.number_of_vocabs += 1
        for word in stemmed3:
            if learn:
                Document.vocabs.put_in(word,30)
                
            self.word_bag.put_in(word,30)
            self.number_of_vocabs += 1

    def __add__(self,other):
        res = Document(Document.vocabs)
        res.word_bag = self.word_bag + other.word_bag    
        return res
    
    def vocabulary_length(self):
        return len(Document.vocabs)
                
    def bag_of_words(self):
        return self.word_bag.get_bag()
        
    def words(self):
        d =  self.word_bag.get_bag()
        return d.keys()
    
    def term_freq(self,word):
        bow =  self.word_bag.get_bag()
        if word in bow:
            return bow[word]
        else:
            return 0
    def get_term_weight(self,word):          
        if word in Document.term_weight:
            return Document.term_weight[word]
        else:
            return 0
    def __and__(self, other):
        intersection = []
        vocabs1 = self.words()
        for word in other.words():
            if word in vocabs1:
                intersection += [word]
        return intersection

# represents the document class
# in our case, there will be two doc classes:
# "pos" and "neg"
class DocumentClass(Document):
    def __init__(self, vocabulary):
        Document.__init__(self, vocabulary)
        self.number_of_docs = 0
        self.SumN = 0;
    def Probability(self,word):
        N = self.term_freq(word) #*self.get_term_weight(word)
        prob = N+0.000001/math.sqrt(self.SumN)+1
        return math.log10(prob)

    def __add__(self,other):
        res = DocumentClass(self.vocabs)
        res.word_bag = self.word_bag + other.word_bag 
        res.number_of_docs = self.number_of_docs + other.number_of_docs
        return res

    def inc_num_doc(self):
        self.number_of_docs +=1

    def set_SumN(self):
        for i in self.words():
            self.SumN += (math.pow(self.term_freq(i),2))

    def num_docs(self):
        return self.number_of_docs
    

class Classifier(object):
    def __init__(self):
        self.classes = {'pos': DocumentClass(BagOfWords()), 
                        'neg': DocumentClass(BagOfWords())}
        self.vocabs = BagOfWords()
        self.number_of_documents = 0 
    
    def sum_vocabs_in_class(self, dclass):
        sum = 0
        for word in self.vocabs.words():
            WaF = self.classes[dclass].bag_of_words()
            if word in WaF:
                sum +=  WaF[word]
        return sum
        
    def learn(self, text, star):
        if (star == 3):     # we dont want to learn from a neutral review 
            return          # because it has mixed pos and neg features
        else:
            category = DocumentClass(self.vocabs)
            doc = Document(self.vocabs)
            doc.read(text, learn=True)
            category += doc
            if (star <= 2):
                self.classes['neg'] += category
            else:
                self.classes['pos'] += category
        self.number_of_documents += 1

    def get_statistics(self):
        print('calculating statistics...')
        #for w in Document.vocabs.words():
        #    for c in self.classes:
        #        temp = 1
        #        if w in self.classes[c].words():
        #             temp += 1
        #             continue
        #    Document.term_weight[w]=math.log10(self.number_of_documents/temp)
        for word in self.classes['pos'].word_bag.get_bag():
            if word in self.classes['neg'].word_bag.get_bag():
                self.classes['pos'].word_bag.get_bag()[word] = 0
                self.classes['neg'].word_bag.get_bag()[word] = 0

        for c in self.classes:
            print("most frequent words in " + c + " are:")
            word_map = self.classes[c].word_bag.get_bag()
            sorted_tw = sorted(word_map.items(),
                               key=operator.itemgetter(1),
                               reverse=True)[:15]
            for tw in sorted_tw:
                print(tw)


        for c in self.classes:
            print(str(c), self.classes[c].num_docs())
            self.classes[c].set_SumN()

    def classify(self, doc, dclass = ""):
        if dclass:
            #sum_dclass = self.sum_vocabs_in_class(dclass)
            prob = 0
        
            d = Document(self.vocabs)
            d.read(doc)
            # for j in self.classes:
            #    sum_j = self.sum_vocabs_in_class(j)
            logprob = 0
            for i in d.words():
                logprob += self.classes[dclass].Probability(i)
              #  wf = self.classes[j].Probability(i)
              #  r = wf * sum_dclass / (wf_dclass * sum_j)
            
#            print(dclass, prod, self.number_of_documents)
            prob = logprob + math.log10(self.classes[dclass].num_docs() / self.number_of_documents)
            return prob
        else:
            prob_list = []
            for dclass in self.classes:
                prob = self.classify(doc, dclass)
                prob_list.append([dclass,prob])
            prob_list.sort(key = lambda x: x[1], reverse = True)
            return prob_list[0][0]


   # def save(self, path):
   #     output = io.open(path,'wb')
   #     output.write('%d' % self.number_of_documents)
   #     for docs in Document.rom __future__ import division, print_function

