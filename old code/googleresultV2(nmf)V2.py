# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 12:49:04 2018

@author: Jimmy
"""

import pandas as pd
import numpy as np
import requests
import re
import glob
from gensim.models.word2vec import Word2Vec
import os

import jieba
import jieba.analyse
from hanziconv import HanziConv

from functools import reduce
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

jieba_path = r'C:\Users\Jimmy\Anaconda3\Lib\site-packages\jieba'

#load jieba dictionary
jieba.set_dictionary(os.path.join(jieba_path, 'dict.txt'))
jieba.analyse.set_stop_words(os.path.join(jieba_path, 'stop_words.txt'))
jieba.load_userdict(os.path.join(jieba_path, 'user_dict.txt'))

list_url = glob.glob(r'C:\Users\Jimmy\Desktop\Python\Googleresult\CCCC\1?.html')
#list_url = [url1,url2,url3]

def read_from_file(list_url):
    google_df = pd.DataFrame(columns = ['heading', 'content', 'google_tag', 'extract_tag'])
    for url in list_url:
        with open(url, 'rb') as f:
            soup = BeautifulSoup(f.read().decode('utf8'), features = "html.parser")
        for articles in soup.find_all('div', class_='g'):
            if not re.search(re.compile('youtube\.com|img|books.google.com'), articles.prettify()):
                heading = articles.find('a').text
                content = articles.find('span', class_='st').text
                #replace date and  ...
                cleanr = re.compile(u'[0-9]{1,2}[ ]?月[0-9]{1,2}[ ]?日|\.{3}|[.]{3}|\\xa0|On\.cc|i-Cable|China Press|\.\.')
                heading = re.sub(cleanr, '', heading)
                content = re.sub(cleanr, '', content)
                google_tag = ''
                for tag in articles.find('div', class_='s').find_all('b')[1:]:
                    if tag != r'...':
                        google_tag = google_tag + ' ' + tag.text
                row = pd.Series([heading,content,google_tag,'0'], index = ['heading', 'content', 'google_tag', 'extract_tag'])
                google_df = google_df.append(row,ignore_index=True)
    return google_df

#creating a DF contain heading, content, extract tag,

def google_content(list_url):
    google_df = pd.DataFrame(columns = ['heading', 'content', 'google_tag', 'extract_tag'])
    for url in list_url:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "lxml")
        for articles in soup.find_all('div', class_='g'):
            if not re.search(re.compile('youtube\.com|img|books.google.com'),articles.prettify()):
                heading = articles.find('a').text
                content = articles.find('span', class_='st').text
                #replace date and  ...
                cleanr = re.compile(u'[0-9]{1,2}[ ]?月[0-9]{1,2}[ ]?日|\.{3}|[.]{3}|\\xa0|On\.cc|i-Cable|China Press|\.\.')
                heading = re.sub(cleanr, '', heading)
                content = re.sub(cleanr, '', content)
                google_tag = ''
                for tag in articles.find('div', class_='s').find_all('b')[1:]:
                    if tag != r'...':
                        google_tag = google_tag + ' ' + tag.text
                row = pd.Series([heading,content,google_tag,'0'], index = ['heading', 'content', 'google_tag', 'extract_tag'])
                google_df = google_df.append(row,ignore_index=True)
    return google_df

# text cutting by jieba, convert to traditional word at the same time
    
def sim_to_trad(df):
    df[['heading', 'content']] = df[['heading','content']].applymap(lambda x : HanziConv.toSimplified(x))
    
def cleaning(df):
    cleanr1 = re.compile(u'[0-9]{1,2}月[0-9]{1,2}日|\.{3}|[0-9] hours ago')
    cleanr2 = re.compile(u'[0-9]{1} day[s]? ago|[0-9]{1,2} hour[s]? ago|[0-9]{1,2} min[s]? ago')
    df[['heading', 'content']] = df[['heading','content']].applymap(lambda x: re.sub(cleanr1, '', x))
    df[['heading', 'content']] = df[['heading','content']].applymap(lambda x: re.sub(cleanr2, 'recent', x))
#    aka = re.compile(u'中國交建|中交建|中交|中國交通建設集團有限公司|中國交通建設')
#    df[['heading','content']] = df[['heading','content']].applymap(lambda x: re.sub(aka, '中交', x))

def extract_tag(df):
    for index in df.index:
        cut_heading = jieba.analyse.extract_tags(HanziConv.toSimplified(df.loc[index,'heading']), topK =50)
        cut_content = jieba.analyse.extract_tags(HanziConv.toSimplified(df.loc[index,'content']), topK =50)
        df.loc[index,'extract_tag'] = reduce(lambda x,y: x + ' ' + y,cut_heading + cut_content)
        df.loc[index,'len_cut_heading'] = len(cut_heading)
        df.loc[index,'len_cut_content'] = len(cut_content)
        #print(df.loc[index,'extract_tag'])

def tfidf(df):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(df['extract_tag'])

def tfidf_2(df):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(df['extract_tag'])

def svd(tfidf):
    svd = TruncatedSVD(n_components=int(tfidf.shape[1]/25), n_iter=7, random_state=42)
    matrix = svd.fit_transform(tfidf)
    print(svd.explained_variance_ratio_.sum())  
    return matrix

def nmf(tfidf):
    model = NMF(n_components=int(60), random_state=42, max_iter=20)
    matrix = model.fit_transform(tfidf)
    return matrix

def KM_clustering_selection(df, matrix):
    KM = KMeans(n_jobs=-1)
    KM = KMeans(random_state=42)
    classifier = GridSearchCV(KM,
                                    {"n_clusters": np.logspace(np.log10(df.shape[0] / 100),
                                                               np.log10(df.shape[0] / 1.5), 5).astype(int)},
                                    n_jobs=-1)
    classifier.fit(matrix)
    df['labels'] = classifier.best_estimator_.labels_
    df['distance_point_to_cluster'] = classifier.best_estimator_.transform(matrix).min(axis=1)
    print("inertia {}".format(classifier.best_estimator_.inertia_))
    print("Number of Cluster {}".format(df['labels'].nunique()))
       
    
def printing_result(df):
    print(df['labels'].nunique())
    for label in df['labels'].value_counts().index[30:40]:
        print('cluster {}'.format(label))
        tag = ''
        for index in df[df['labels'] == label].index:
            tag = tag + df.loc[index,'heading'] + df.loc[index,'content']
        print(google_df[google_df['labels'] == label]['distance_point_to_cluster'].mean())
        print(jieba.analyse.extract_tags(HanziConv.toSimplified(tag),topK =10))
        print(df['content'][df['labels'] == label])

def printing_result2(df):
    print(df['labels'].nunique())
    for label in df['labels'].value_counts().index[0:10]:
        print('cluster {}'.format(label))
        print(df['extract_tag'][df['labels'] == label])
                
def printing_result3(clustered_df, df):
    print(clustered_df['labels'].nunique())
    for index in clustered_df['labels'].value_counts().index[0:3]:
        print('{} largest cluster'.format(index))
        for label in clustered_df[clustered_df['labels'] == index].index:
            print('cluster {}'.format(label))
            print(df.loc[df['labels'] == label,['content']])
        
def wordtovec(df):
    df['cut_text'] = google_df['heading'] + google_df['content']
    df['cut_text'] = df['cut_text'].apply(lambda x: re.sub(r"【】[|\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", x))
    df['cut_text'] = df['cut_text'].apply(lambda x: jieba.lcut(x))

def cluster_cluster(df):
    cluster = pd.DataFrame(columns=['extract_tag'])
    for label in df['labels'].value_counts().index:
        tag = ''
        for index in df[df['labels'] == label].index:
            tag = tag + df.loc[index,'heading'] + df.loc[index,'content']
        cluster.loc[label,'extract_tag'] = reduce(lambda x, y: x+' '+y,jieba.analyse.extract_tags(HanziConv.toSimplified(tag),topK =50))
    return cluster


google_df = read_from_file(list_url)
sim_to_trad(google_df)
cleaning(google_df)
extract_tag(google_df)
#google_tfidf = tfidf(google_df)
google_tfidf = tfidf_2(google_df)
#matrix = svd(google_tfidf)
matrix = nmf(google_tfidf)
KM_clustering_selection(google_df,matrix)
printing_result(google_df)

temp1 = google_df[['heading','labels']].groupby('labels').count()

cluster_df = cluster_cluster(google_df)
cluster_tfidf= tfidf_2(cluster_df)
cluster_matrix = nmf(cluster_tfidf)
KM_clustering_selection(cluster_df,cluster_matrix)
printing_result2(cluster_df)

cluster_df = cluster_df.sort_index()
cluster_df['number_of_article'] = temp1
temp2 = cluster_df[['labels','number_of_article']].groupby('labels').sum()
temp2 = temp2.sort_values('number_of_article', ascending=False)

printing_result3(cluster_df, google_df)





google_df.loc[270,:]
google_df.loc[270,'heading']
google_df.loc[270,'content']

google_df.loc[google_df['labels'] == 1,:]


'''
wordtovec(google_df)
model = Word2Vec(google_df['cut_text'])
model.wv.most_similar('京津冀', topn=10)

google_df['distance_point_to_cluster'].plot(kind='hist')
plt.show()
plt.hist(google_tfidf.sum(axis=1))
plt.show()

stop_word = re.compile(u'(?<= ).{1,3}網(?= )')

for tag in google_df['extract_tag']:
    print(re.search(stop_word, tag))
'''
