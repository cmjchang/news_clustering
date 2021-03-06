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

import jieba
import jieba.analyse
from hanziconv import HanziConv

from functools import reduce
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

'''
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES


'''

#load jieba dictionary
jieba.set_dictionary('Anaconda3/Lib/site-packages/jieba/dict.txt.big')
jieba.analyse.set_stop_words(r"C:/Users/Jimmy/Anaconda3/Lib/site-packages/jieba/stop_words.txt")
jieba.load_userdict("Anaconda3/Lib/site-packages/jieba/user_dict.txt")

# Google url
url1 = r'https://www.google.com.hk/search?num=100&ei=UpI3W_jEBdf4hwPVxJvgBg&q=%22%E4%B8%AD%E5%9B%BD%E8%88%B9%E8%88%B6%22+AND+(%E6%B6%89%E5%AB%8C+OR+%E6%8B%98+OR+%E8%B4%AA%E6%B1%A1+OR+%E8%B4%BF+OR+%E4%BE%B5%E5%90%9E+OR+%E5%8F%8D%E8%B4%AA+OR+%E6%B4%97%E9%92%B1+OR+%E6%8C%AA%E7%94%A8+OR+%E8%85%90%E8%B4%A5+OR+%E7%A7%81%E5%90%9E+OR+%E7%A7%81%E8%97%8F+OR+%E5%85%AC%E5%AE%89+OR+%E8%AD%A6%E6%96%B9+OR+%E5%8F%8C%E8%A7%84+OR+%E9%80%AE%E6%8D%95+OR+%E5%88%A4%E5%86%B3+OR+%E5%88%91+OR+%E8%A2%AB%E5%91%8A+OR+%E8%A2%AB%E6%8E%A7+OR+%E8%A2%AB%E5%88%A4+OR+%E8%AF%89%E8%AE%BC+OR+%E8%B5%B7%E8%AF%89+OR+%E8%A3%81%E5%88%A4+OR+%E8%A3%81%E5%86%B3)&oq=%22%E4%B8%AD%E5%9B%BD%E8%88%B9%E8%88%B6%22+AND+(%E6%B6%89%E5%AB%8C+OR+%E6%8B%98+OR+%E8%B4%AA%E6%B1%A1+OR+%E8%B4%BF+OR+%E4%BE%B5%E5%90%9E+OR+%E5%8F%8D%E8%B4%AA+OR+%E6%B4%97%E9%92%B1+OR+%E6%8C%AA%E7%94%A8+OR+%E8%85%90%E8%B4%A5+OR+%E7%A7%81%E5%90%9E+OR+%E7%A7%81%E8%97%8F+OR+%E5%85%AC%E5%AE%89+OR+%E8%AD%A6%E6%96%B9+OR+%E5%8F%8C%E8%A7%84+OR+%E9%80%AE%E6%8D%95+OR+%E5%88%A4%E5%86%B3+OR+%E5%88%91+OR+%E8%A2%AB%E5%91%8A+OR+%E8%A2%AB%E6%8E%A7+OR+%E8%A2%AB%E5%88%A4+OR+%E8%AF%89%E8%AE%BC+OR+%E8%B5%B7%E8%AF%89+OR+%E8%A3%81%E5%88%A4+OR+%E8%A3%81%E5%86%B3)&aqs=chrome..69i57.7953j0j1&sourceid=chrome&ie=UTF-8'
url2 = r'https://www.google.com.hk/search?num=100&ei=UpI3W_jEBdf4hwPVxJvgBg&q=%22%E4%B8%AD%E5%9B%BD%E8%88%B9%E8%88%B6%22+AND+%28%E8%AD%A6%E5%91%8A+OR+%E4%BB%B2%E8%A3%81+OR+%E7%8B%B1+OR+%E7%BD%9A+OR+%E7%BD%AA+OR+%E7%BA%A0%E7%BA%B7+OR+%E8%B5%94%E5%81%BF+OR+%E6%8C%87%E6%8E%A7+OR+%E4%BD%9C%E5%81%87+OR+%E9%80%A0%E5%81%87+OR+%E8%B4%A8%E7%96%91+OR+%E7%9E%92%E7%A8%8E+OR+%E6%AC%A0%E7%A8%8E+OR+%E9%AA%97+OR+%E6%AC%BA%E8%AF%88+OR+%E9%93%B6%E7%9B%91+OR+%E8%AF%81%E7%9B%91+OR+%E5%AE%A1%E8%AE%A1%E7%BD%B2+OR+%E8%BF%9D%E5%8F%8D+OR+%E8%BF%9D%E8%A7%84+OR+%E8%BF%9D%E6%B3%95+OR+%E8%BF%9D%E7%A6%81+OR+%E8%BF%9D%E4%BE%8B+OR+%E4%BC%AA%E9%80%A0+OR+%E5%A4%84%E7%BD%9A%29&oq=%22%E4%B8%AD%E5%9B%BD%E8%88%B9%E8%88%B6%22+AND+%28%E8%AD%A6%E5%91%8A+OR+%E4%BB%B2%E8%A3%81+OR+%E7%8B%B1+OR+%E7%BD%9A+OR+%E7%BD%AA+OR+%E7%BA%A0%E7%BA%B7+OR+%E8%B5%94%E5%81%BF+OR+%E6%8C%87%E6%8E%A7+OR+%E4%BD%9C%E5%81%87+OR+%E9%80%A0%E5%81%87+OR+%E8%B4%A8%E7%96%91+OR+%E7%9E%92%E7%A8%8E+OR+%E6%AC%A0%E7%A8%8E+OR+%E9%AA%97+OR+%E6%AC%BA%E8%AF%88+OR+%E9%93%B6%E7%9B%91+OR+%E8%AF%81%E7%9B%91+OR+%E5%AE%A1%E8%AE%A1%E7%BD%B2+OR+%E8%BF%9D%E5%8F%8D+OR+%E8%BF%9D%E8%A7%84+OR+%E8%BF%9D%E6%B3%95+OR+%E8%BF%9D%E7%A6%81+OR+%E8%BF%9D%E4%BE%8B+OR+%E4%BC%AA%E9%80%A0+OR+%E5%A4%84%E7%BD%9A%29&gs_l=psy-ab.3...4766.4766.0.5189.1.1.0.0.0.0.0.0..0.0....0...1c.1.64.psy-ab..1.0.0....0.5Wt83dKPHtA'
url3 = r'https://www.google.com.hk/search?num=100&ei=W5I3W6X6CYnNvgTI45zYCw&q=%22%E4%B8%AD%E5%9B%BD%E8%88%B9%E8%88%B6%22+AND+%28%E8%99%9A%E5%81%87+OR+%E9%80%BE%E6%9C%9F+OR+%E5%86%85%E5%B9%95+OR+%E6%93%8D%E7%BA%B5+OR+%E6%93%8D%E6%8E%A7+OR+%E5%A4%84%E5%88%86+OR+%E7%96%8F%E5%BF%BD+OR+%E4%BA%8B%E6%95%85+OR+%E6%84%8F%E5%A4%96+OR+%E4%BE%B5%E7%8A%AF+OR+%E4%BE%B5%E6%9D%83+OR+%E7%9B%97+OR+%E6%8A%95%E8%AF%89+OR+%E8%AF%AF%E5%AF%BC+OR+%E7%BB%B4%E6%9D%83+OR+%E6%8B%96%E6%AC%A0+OR+%E5%BF%BD%E6%82%A0+OR+%E6%94%B6%E4%B9%B0+OR+%E4%B8%91%E9%97%BB+OR+%E7%A0%B4%E4%BA%A7+OR+%E6%BD%9C%E8%A7%84%E5%88%99+OR+%E8%BF%9D%E7%BA%A6+OR+%E9%9D%9E%E6%B3%95+OR+%E6%B1%A1%E6%9F%93%29&oq=%22%E4%B8%AD%E5%9B%BD%E8%88%B9%E8%88%B6%22+AND+%28%E8%99%9A%E5%81%87+OR+%E9%80%BE%E6%9C%9F+OR+%E5%86%85%E5%B9%95+OR+%E6%93%8D%E7%BA%B5+OR+%E6%93%8D%E6%8E%A7+OR+%E5%A4%84%E5%88%86+OR+%E7%96%8F%E5%BF%BD+OR+%E4%BA%8B%E6%95%85+OR+%E6%84%8F%E5%A4%96+OR+%E4%BE%B5%E7%8A%AF+OR+%E4%BE%B5%E6%9D%83+OR+%E7%9B%97+OR+%E6%8A%95%E8%AF%89+OR+%E8%AF%AF%E5%AF%BC+OR+%E7%BB%B4%E6%9D%83+OR+%E6%8B%96%E6%AC%A0+OR+%E5%BF%BD%E6%82%A0+OR+%E6%94%B6%E4%B9%B0+OR+%E4%B8%91%E9%97%BB+OR+%E7%A0%B4%E4%BA%A7+OR+%E6%BD%9C%E8%A7%84%E5%88%99+OR+%E8%BF%9D%E7%BA%A6+OR+%E9%9D%9E%E6%B3%95+OR+%E6%B1%A1%E6%9F%93%29&gs_l=psy-ab.3...1402.1402.0.2335.1.1.0.0.0.0.0.0..0.0....0...1c.1.64.psy-ab..1.0.0....0.0sjecp9nel4'

list_url = glob.glob(r'C:\Users\Jimmy\Desktop\Python\Googleresult\CCCC\*.html')
#list_url = [url1,url2,url3]


url = r'C:\Users\Jimmy\Desktop\Python\Googleresult\CCCC\22.html'


def read_from_file(list_url):
    google_df = pd.DataFrame(columns = ['heading', 'content', 'google_tag', 'extract_tag'])
    for url in list_url:
        with open(url, 'rb') as f:
            soup = BeautifulSoup(f.read().decode('utf8'),'lxml')
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
    df[['heading','content']] = df[['heading','content']].applymap(lambda x : HanziConv.toTraditional(x))
    
def cleaning(df):
    cleanr1 = re.compile(u'[0-9]{1,2}月[0-9]{1,2}日|\.{3}|[0-9] hours ago')
    cleanr2 = re.compile(u'[0-9]{1} day[s]? ago|[0-9]{1,2} hour[s]? ago|[0-9]{1,2} min[s]? ago')
    df[['heading','content']] = df[['heading','content']].applymap(lambda x: re.sub(cleanr1, '', x))
    df[['heading','content']] = df[['heading','content']].applymap(lambda x: re.sub(cleanr2, 'recent', x))
#    aka = re.compile(u'中國交建|中交建|中交|中國交通建設集團有限公司|中國交通建設')
#    df[['heading','content']] = df[['heading','content']].applymap(lambda x: re.sub(aka, '中交', x))

def extract_tag(df):
    for index in df.index:
        cut_heading = jieba.analyse.extract_tags(HanziConv.toTraditional(df.loc[index,'heading']), topK =50)
        cut_content = jieba.analyse.extract_tags(HanziConv.toTraditional(df.loc[index,'content']), topK =50)
        df.loc[index,'extract_tag'] = reduce(lambda x,y: x + ' ' + y,cut_heading + cut_content)
        df.loc[index,'len_cut_heading'] = len(cut_heading)
        df.loc[index,'len_cut_content'] = len(cut_content)
        #print(df.loc[index,'extract_tag'])

def tfidf(df):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(df['extract_tag'])

def svd(tfidf):
    svd = TruncatedSVD(n_components=int(tfidf.shape[1]/25), n_iter=7, random_state=42)
    matrix = svd.fit_transform(tfidf)
    print(svd.explained_variance_ratio_.sum())  
    return matrix

def KM_clustering_selection(df, matrix):
    KM = KMeans(n_jobs=-1)
    KM = KMeans(random_state=42)
    classifier = GridSearchCV(KM,
                                    {"n_clusters": np.logspace(np.log10(df.shape[0] / 100),
                                                               np.log10(df.shape[0] / 1.5), 3).astype(int)},
                                    n_jobs=-1)
    classifier.fit(matrix)
    df['labels'] = classifier.best_estimator_.labels_
    df['distance_point_to_cluster'] = classifier.best_estimator_.transform(matrix).min(axis=1)
    print("inertia {}".format(classifier.best_estimator_.inertia_))
    print("Number of Cluster {}".format(df['labels'].nunique()))
       
    
def printing_result(df):
    print(df['labels'].nunique())
    for label in df['labels'].value_counts().index[0:20]:
        print('cluster {}'.format(label))
        tag = ''
        for index in df[df['labels'] == label].index:
            tag = tag + df.loc[index,'heading'] + df.loc[index,'content']
        print(google_df[google_df['labels'] == label]['distance_point_to_cluster'].mean())
        print(jieba.analyse.extract_tags(HanziConv.toTraditional(tag),topK =10))
        print(df['content'][df['labels'] == label])

def printing_result2(df):
    print(df['labels'].nunique())
    for label in df['labels'].value_counts().index[0:10]:
        print('cluster {}'.format(label))
        print(df['extract_tag'][df['labels'] == label])
        
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
        cluster.loc[label,'extract_tag'] = reduce(lambda x, y: x+' '+y,jieba.analyse.extract_tags(HanziConv.toTraditional(tag),topK =50))
    return cluster


        
'''
def svd_dbscan_clustering(df, tfidf, n_clusters = 20):
    svd = TruncatedSVD(n_components=np.int(tfidf.shape[1]/25), n_iter=7, random_state=42)
    matrix = svd.fit_transform(tfidf)
    print(svd.explained_variance_ratio_.sum())  
    AC = DBSCAN(eps=20, min_samples =5)
    AC.fit(matrix)
    labels = AC.labels_
    len(labels)
    df['labels'] = labels
    for label in df['labels'].value_counts().index:
        print('cluster {}'.format(label))
        print(df['content'][df['labels'] == label])
'''

'''
# sample for cluster analysis (represented by list)
sample = read_sample(SIMPLE_SAMPLES.SAMPLE_SIMPLE4)

# create object of X-Means algorithm that uses CCORE for processing
# initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
# let's avoid random initial centers and initialize them using K-Means++ method:
initial_centers = kmeans_plusplus_initializer(matrix, 19).initialize()
xmeans_instance = xmeans(matrix, initial_centers);
xmeans_instance.process()
clusters = xmeans_instance.get_clusters()
len(clusters)
for x in clusters:
    print(google_df.loc[x,'content'])
'''

#google_df =google_content(list_url)


google_df = read_from_file(list_url)
sim_to_trad(google_df)
cleaning(google_df)
extract_tag(google_df)
google_tfidf= tfidf(google_df)
matrix = svd(google_tfidf)
KM_clustering_selection(google_df,matrix)
printing_result(google_df)

cluster_df = cluster_cluster(google_df)
cluster_tfidf= tfidf(cluster_df)
cluster_matrix = svd(cluster_tfidf)
KM_clustering_selection(cluster_df,cluster_matrix)
printing_result2(cluster_df)

wordtovec(google_df)
model = Word2Vec(google_df['cut_text'])
model.wv.most_similar('京津冀', topn=10)

google_df['distance_point_to_cluster'].plot(kind='hist')
plt.hist(google_tfidf.sum(axis=1))

stop_word = re.compile(u'(?<= ).{1,3}網(?= )')

for tag in google_df['extract_tag']:
    print(re.search(stop_word, tag))

google_df.loc[(google_df['labels'] == 1310),['heading','content','extract_tag']]
google_df.loc[(google_df['labels'] == 33),['heading','content','extract_tag']]


cluster_df.loc[(cluster_df['labels'] == 1),['extract_tag']]

google_df.loc[4,'content']

'''

stop_word = re.compile(u'(?<= ).{1,3}網(?= )')

for tag in google_df['extract_tag']:
    print(re.search(stop_word, tag))


google_df['len_cut_heading'].describe()
google_df['len_cut_content'].describe()

google_df.loc[20,'heading']

cluster_tag = cluster_cluster(google_df)
cluster_tfidf= tfidf(cluster_tag)
matrix = svd(cluster_tfidf)
KM_clustering_selection(cluster_tag,matrix)
printing_result2(cluster_tag)

for record in google_df.index:
    print (google_df.loc[record,'extract_tag'])

for num in [102]:
    for record in google_df[google_df['labels'] == num].index:
        print (google_df.loc[record,'content'])
    
google_df.loc[(google_df['labels'] == 93),['heading','content']]
google_df.loc[(google_df['labels'] == 46),['heading','content']]
google_df.loc[(google_df['labels'] == 38),['heading','content']]


jieba.set_dictionary('Anaconda3/Lib/site-packages/jieba/dict.txt.big')
jieba.load_userdict("Anaconda3/Lib/site-packages/jieba/user_dict.txt")
jieba.analyse.set_stop_words(r"C:/Users/Jimmy/Anaconda3/Lib/site-packages/jieba/stop_words.txt")
text = '東網'
jieba.analyse.extract_tags(text, topK =10)



'''
AC = AgglomerativeClustering(n_clusters=12)
AC.fit(matrix)
labels = AC.labels_
AC.n_leaves_
AC.n_components_
AC.children_ 
'''
'''
evaluation = pd.DataFrame(columns=['n_clusters','inertia'])
for n_clusters in np.arange(10,40,2):
    KM = KMeans(n_clusters = n_clusters, random_state=42)
    KM.fit(matrix)
    row = pd.Series([n_clusters,KM.inertia_], index = ['n_clusters', 'inertia'])
    evaluation = evaluation.append(row,ignore_index=True)    
    print("n_cluster = {}, inertia = {}".format(n_clusters,KM.inertia_))
    print("n_cluster = {}, customized_inertia = {}".format(n_clusters,KM.inertia_ + n_clusters**2))

KM = KMeans(n_clusters = 16, random_state=42)
evaluation.plot(kind='line', )
''' 
    


'''
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(google_df['extract_tag'])
temp_df = pd.DataFrame(columns=['word','wordcount'])
temp_df['word'] = vectorizer.get_feature_names()
temp_df['wordcount'] = matrix.sum(axis=0).reshape(-1,1)
temp_df.sort_values('wordcount')

np.sum(matrix.A,axis=1)
np.sum((matrix.A)**2,axis=1)
np.count_nonzero(matrix.A,axis=1)

vector_length = np.power(np.sum((matrix.A)**2,axis=1), 1.0 / np.count_nonzero(matrix.A,axis=1))
vector_length.max()
vector_length.min()
'''
vectorizer = CountVectorizer()
tfidf = vectorizer.fit_transform(google_df['extract_tag'])
vectorizer.vocabulary_
temp_df = pd.DataFrame(columns=['word','wordcount'])
temp_df['word'] = vectorizer.get_feature_names()
temp_df['wordcount'] = tfidf.sum(axis=0).reshape(-1,1)
temp_df.sort_values('wordcount') 

svd = TruncatedSVD(n_components=np.int(tfidf.shape[1]/25), n_iter=7, random_state=42)
matrix = svd.fit_transform(tfidf)
print(svd.explained_variance_ratio_.sum())  

KM = KMeans(n_clusters = 9 ,n_jobs=-1, random_state=42)
KM.fit(matrix)
KM.fit_transform(matrix).shape
KM.labels_
KM.cluster_centers_
KM.inertia_
KM.fit_predict(matrix)

KM_random = RandomizedSearchCV(KM,
                   {'n_clusters': list(np.arange(5,41,2))})
KM_random.get_params().keys()
KM_random.fit(matrix)
KM_random.best_estimator_ 

google_df.shape
matrix.shape
google_df['distance_point_to_cluster'] = KM_random.best_estimator_.transform(matrix).min(axis=1)
google_df['labels'] = KM_random.best_estimator_.labels_

for label in google_df['labels'].value_counts().index:
    print('cluster {}'.format(label))
    tag = ''
    for index in google_df[google_df['labels'] == label].index:
        tag = tag + google_df.loc[index,'heading'] + google_df.loc[index,'content']
    print(google_df[google_df['labels'] == label]['distance_point_to_cluster'].mean())
    print(jieba.analyse.extract_tags(HanziConv.toTraditional(tag),topK =10))
    print(google_df[['content']][google_df['labels'] == label])

for record in google_df.index:
    print(google_df['extract_tag'][record])
'''