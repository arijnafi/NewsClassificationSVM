# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""  

from pathlib import Path
import numpy as np
import glob
import os
import pandas as pd
from bs4 import BeautifulSoup as bs
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path1 = Path("datasetSTKI/ekonomi/") 
path2 = Path("datasetSTKI/politik/")
path3 = Path("datasetSTKI/olahraga/")
path4 = Path("datasetSTKI/entertainment/")     

label = []
artikel = []
dataset = []

#file XML Ekonomi
for filename in glob.glob(os.path.join(path1, "*.xml")):
    with open(filename, 'r', encoding="utf8") as open_file:
        content = open_file.read()
        soup = bs(content,"lxml")
        isi = soup.find("isi").get_text()
#case folding
        lowercase = isi.lower()
#symbol removal
        rmvsym = re.sub(r'[^\w\s]', '', lowercase)
#stopword removal      
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        stoprmv = stopword.remove(str(rmvsym))
#Tokenization
        tokens = word_tokenize(stoprmv)
#stemming
        stemFactory = StemmerFactory()
        stemmer = stemFactory.create_stemmer()
        stemmed = ""
        for word in tokens:
            root = stemmer.stem(word)
            stemmed += root + " "           
#masukin ke array
        label.append("eko")
        artikel.append(stemmed)   
        
        
#file XML Politik
for filename in glob.glob(os.path.join(path2, "*.xml")):
    with open(filename, 'r', encoding="utf8") as open_file:
        content = open_file.read()
        soup = bs(content,"lxml")
        isi = soup.find("isi").get_text()
#case folding
        lowercase = isi.lower()
#symbol removal
        rmvsym = re.sub(r'[^\w\s]', '', lowercase)
#stopword removal      
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        stoprmv = stopword.remove(str(rmvsym))
#Tokenization
        tokens = word_tokenize(stoprmv)
#stemming
        stemFactory = StemmerFactory()
        stemmer = stemFactory.create_stemmer()
        stemmed = ""
        for word in tokens:
            root = stemmer.stem(word)
            stemmed += root + " "           
#masukin ke array
        label.append("pol")
        artikel.append(stemmed)   
        
        
#file XML Olahraga
for filename in glob.glob(os.path.join(path3, "*.xml")):
    with open(filename, 'r', encoding="utf8") as open_file:
        content = open_file.read()
        soup = bs(content,"lxml")
        isi = soup.find("isi").get_text()
#case folding
        lowercase = isi.lower()
#symbol removal
        rmvsym = re.sub(r'[^\w\s]', '', lowercase)
#stopword removal      
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        stoprmv = stopword.remove(str(rmvsym))
#Tokenization
        tokens = word_tokenize(stoprmv)
#stemming
        stemFactory = StemmerFactory()
        stemmer = stemFactory.create_stemmer()
        stemmed = ""
        for word in tokens:
            root = stemmer.stem(word)
            stemmed += root + " "          
#masukin ke array
        label.append("olg")
        artikel.append(stemmed)   
        

#file XML Entertainment
for filename in glob.glob(os.path.join(path4, "*.xml")):
    with open(filename, 'r', encoding="utf8") as open_file:
        content = open_file.read()
        soup = bs(content,"lxml")
        isi = soup.find("isi").get_text()
#case folding
        lowercase = isi.lower()
#symbol removal
        rmvsym = re.sub(r'[^\w\s]', '', lowercase)
#stopword removal      
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        stoprmv = stopword.remove(str(rmvsym))
#Tokenization
        tokens = word_tokenize(stoprmv)
#stemming
        stemFactory = StemmerFactory()
        stemmer = stemFactory.create_stemmer()
        stemmed = ""
        for word in tokens:
            root = stemmer.stem(word)
            stemmed += root + " "
#masukin ke array
        label.append("ent")
        artikel.append(stemmed)        

dataset = [label, artikel]

dtype = [('Col1','float32'), ('Col2','float32')]
#values = numpy.zeros(20, dtype=dtype)
index = [str(i) for i in range(1, len(dataset)+1)]

df = pd.DataFrame(dataset, index=index)

df_transposed = df.T

df_copy = df_transposed['2'].copy()

#Tfidf        
vectorizer = TfidfVectorizer()
artikel_mat = vectorizer.fit_transform(df_copy) 
        
#split data
berita_train, berita_test, label_train, label_test = train_test_split(artikel_mat, df_transposed['1'], test_size=0.25, random_state=20)


svm_model= svm.SVC(kernel='linear')
svm_model.fit(berita_train, label_train)
svm_y_pred=svm_model.predict(berita_test)

score = accuracy_score(label_test, svm_y_pred)

from sklearn import metrics
csvm = metrics.confusion_matrix(label_test, svm_y_pred)
print(csvm)

from sklearn.metrics import classification_report
y_true_svm = label_test
y_pred_svm = svm_y_pred
print(classification_report(y_true_svm, y_pred_svm))





