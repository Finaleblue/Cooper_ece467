from __future__ import division
from nltk.corpus import stopwords 
from nltk import tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
import re, os, io, math

class BagOfWords(object):
    def __init__(self):
        self.number_of_vocabs = 0
        self.bag_of_words = {}
        
    def __add__(self,other):
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
        
    def put_in(self,word):
        self.number_of_vocabs += 1
        if word in self.bag_of_words:
            self.bag_of_words[word] += 1
        else:
            self.bag_of_words[word] = 1
    
    def len(self):
        return len(self.bag_of_words)
    
    def words(self):
        return self.bag_of_words.keys()
    
        
    def get_bag(self):
        return self.bag_of_words
        
    def term_freq(self,word):
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

    def read(self,filename, learn=False):
        text = io.open(filename,'r', encoding = 'utf-8').read()
        st = PorterStemmer()
        tknzr = tokenize.RegexpTokenizer(r'\w+')
        vocabs = tknzr.tokenize(text)
        for word in vocabs:
            if word in Document.stops:
                vocabs.remove(word)
        stemmed = []
        for v in vocabs:
            stemmed.append(st.stem(v))
        self.number_of_vocabs = 0
        #vocabs = list(ngrams(vocabs,2))
        #print(vocabs[2])
        for word in stemmed:
            self.word_bag.put_in(word)
            self.number_of_vocabs += 1
            if learn:
                Document.vocabs.put_in(word)
         
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

class DocumentClass(Document):
    def __init__(self, vocabulary):
        Document.__init__(self, vocabulary)
        self.number_of_docs = 0
        self.SumN = 0;
    def Probability(self,word):
        N = self.term_freq(word) *self.get_term_weight(word)
        prob = (1+N)/math.sqrt(self.SumN)
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
            self.SumN +=math.pow(self.term_freq(i),2)*self.get_term_weight(i)+1

    def num_docs(self):
        return self.number_of_docs
    
class Classifier(object):
    def __init__(self):
        self.classes = {}
        self.vocabs = BagOfWords()
        self.number_of_documents = 0 
    
    def sum_vocabs_in_class(self, dclass):
        sum = 0
        for word in self.vocabs.words():
            WaF = self.classes[dclass].bag_of_words()
            if word in WaF:
                sum +=  WaF[word]
        return sum
        
    def learn(self, filedir, class_name):
        c = DocumentClass(self.vocabs)
        d = Document(self.vocabs)
        print(filedir +" " + class_name)
        d.read(filedir, learn = True)
        c += d
        if (class_name not in self.classes):
            self.classes[class_name] = c
        else:
            self.classes[class_name] += c
        self.number_of_documents += 1
    def get_statistics(self):
        print('calculating statistics...')
        for w in Document.vocabs.words():
            for c in self.classes:
                temp = 1
                if w in self.classes[c].words():
                     temp += 1
                     continue
            Document.term_weight[w]=math.log10(self.number_of_documents/temp)
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

