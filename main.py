# -*- coding: utf-8 -*-
import pandas as pd 
from nltk.corpus import stopwords
import re
from wordcloud import *
import pylab as plt
from nltk.stem.snowball import SnowballStemmer
from nltk import WordNetLemmatizer 
import sys
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger 

revs = open('reviews.txt', 'r').read()

"""Cleaning"""
revs = revs.lower()
revs = re.sub(r'[-./?!,":;()\']',' ', revs)
revs = re.sub('[-|0-9]',' ', revs)
revs = revs.split()
stwds = stopwords.words('english')

owds = [w for w in revs if w not in stwds]

"""Creating wordcloud"""
wdcloud = WordCloud(width=1000, height=500).generate(' '.join(owds))
plt.figure(figsize=(15,8))
plt.imshow(wdcloud)
plt.axis('off')
plt.show()

"""Stemming and lemmatization using Snowball algo"""
#Reduce words to their root
stemmer = SnowballStemmer("english")
nwds = []
for w in owds:
	try:
		nwds.append(stemmer.stem(w))
	except:
		pass
lmtz = WordNetLemmatizer()
wds = []
for w in nwds:
    try:
        wds.append(lmtz.lemmatize(w))
    except:
        pass

"""Using the Stanford Named Entity Recognizer"""
st = StanfordNERTagger('./lib/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz','./lib/stanford-ner/stanford-ner.jar')
swds = st.tag(owds)
