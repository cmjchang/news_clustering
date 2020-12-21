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
import os

import jieba
import jieba.analyse
from hanziconv import HanziConv

from functools import reduce
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

# jieba_path = r'C:\Users\Jimmy\Anaconda3\Lib\site-packages\jieba'

# load jieba dictionary
# jieba.set_dictionary(os.path.join(jieba_path, 'dict.txt'))

list_url = glob.glob('CCCC/1?.html')


def read_from_file(list_url):
    google_df = pd.DataFrame(columns=['heading', 'content', 'google_tag', 'extract_tag'])
    for url in list_url:
        with open(url, 'rb') as f:
            soup = BeautifulSoup(f.read().decode('utf8'), features="html.parser")
        for articles in soup.find_all('div', class_='g'):
            if not re.search(re.compile('youtube\.com|img|books.google.com'), articles.prettify()):
                heading = articles.find('a').text
                content = articles.find('span', class_='st').text
                # replace date and  ...
                cleanr = re.compile(
                    u'[0-9]{1,2}[ ]?月[0-9]{1,2}[ ]?日|\.{3}|[.]{3}|\\xa0|On\.cc|i-Cable|China Press|\.\.')
                heading = re.sub(cleanr, '', heading)
                content = re.sub(cleanr, '', content)
                google_tag = ''
                for tag in articles.find('div', class_='s').find_all('b')[1:]:
                    if tag != r'...':
                        google_tag = google_tag + ' ' + tag.text
                row = pd.Series([heading, content, google_tag, '0'],
                                index=['heading', 'content', 'google_tag', 'extract_tag'])
                google_df = google_df.append(row, ignore_index=True)
    return google_df


def google_content(list_url):
    # creating a DF contain heading, content, extract tag
    google_df = pd.DataFrame(columns=['heading', 'content', 'google_tag', 'extract_tag'])
    for url in list_url:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "lxml")
        for articles in soup.find_all('div', class_='g'):
            if not re.search(re.compile('youtube\.com|img|books.google.com'), articles.prettify()):
                heading = articles.find('a').text
                content = articles.find('span', class_='st').text
                # replace date and  ...
                cleanr = re.compile(
                    u'[0-9]{1,2}[ ]?月[0-9]{1,2}[ ]?日|\.{3}|[.]{3}|\\xa0|On\.cc|i-Cable|China Press|\.\.')
                heading = re.sub(cleanr, '', heading)
                content = re.sub(cleanr, '', content)
                google_tag = ''
                for tag in articles.find('div', class_='s').find_all('b')[1:]:
                    if tag != r'...':
                        google_tag = google_tag + ' ' + tag.text
                row = pd.Series([heading, content, google_tag, '0'],
                                index=['heading', 'content', 'google_tag', 'extract_tag'])
                google_df = google_df.append(row, ignore_index=True)
    return google_df


def sim_to_trad(df):
    df[['heading', 'content']] = df[['heading', 'content']].applymap(lambda x: HanziConv.toSimplified(x))
    return df


def cleaning(df):
    cleanr1 = re.compile(u'[0-9]{1,2}月[0-9]{1,2}日|\.{3}|[0-9] hours ago')
    cleanr2 = re.compile(u'[0-9]{1} day[s]? ago|[0-9]{1,2} hour[s]? ago|[0-9]{1,2} min[s]? ago')
    df[['heading', 'content']] = df[['heading', 'content']].applymap(lambda x: re.sub(cleanr1, '', x))
    df[['heading', 'content']] = df[['heading', 'content']].applymap(lambda x: re.sub(cleanr2, 'recent', x))
    return df


def drop_duplicates(df):
    df = df.drop_duplicates(['heading', 'content'])
    return df


def extract_tag(df):
    for index in df.index:
        cut_heading = jieba.analyse.extract_tags(HanziConv.toSimplified(df.loc[index, 'heading']), topK=50)
        cut_content = jieba.analyse.extract_tags(HanziConv.toSimplified(df.loc[index, 'content']), topK=50)
        df.loc[index, 'extract_tag'] = reduce(lambda x, y: x + ' ' + y, cut_heading + cut_content)
        df.loc[index, 'len_cut_heading'] = len(cut_heading)
        df.loc[index, 'len_cut_content'] = len(cut_content)
        # print(df.loc[index,'extract_tag'])


def vectorizer(df, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(df['extract_tag'])

    if method == 'count':
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(df['extract_tag'])


def dimension_reduction(tfidf, method='nmf', n_components=60, random_state=42):
    if method == 'nmf':
        model = NMF(n_components=n_components, random_state=random_state, max_iter=20)
        matrix = model.fit_transform(tfidf)
        return matrix

    if method == 'svd':
        svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=random_state)
        matrix = svd.fit_transform(tfidf)
        return matrix

    if method == 'tsne':
        model = TSNE(n_components=n_components, n_jobs=-1, method='exact', random_state=random_state)
        matrix = model.fit_transform(tfidf)
        return matrix


def KM_clustering_selection(df, matrix):
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


def KM_clustering_selection_sil_score(df, matrix):
    list_cluster_trial = np.logspace(np.log10(df.shape[0] / 100), np.log10(df.shape[0] / 1.5), 100).astype(int)
    list_cluster_trial = range(50, 201)

    sil_score_list = []
    for ncluster in list_cluster_trial:
        kmean = KMeans(n_clusters=ncluster, n_jobs=-1, random_state=42)
        kmean.fit(matrix)

        sil_score = silhouette_score(matrix, kmean.labels_)
        print('number of clusters: {} and corresponding silhouette score: {:.3f}'.format(ncluster, sil_score))
        sil_score_list.append(sil_score)

    sil_score_df = pd.DataFrame()
    sil_score_df['number_cluster'] = list_cluster_trial
    sil_score_df['silhouette_score'] = sil_score_list
    return sil_score_df


def KM_clustering(df, matrix, ncluster):
    KM = KMeans(n_clusters=ncluster, n_jobs=-1, random_state=42)
    KM.fit(matrix)
    df['labels'] = KM.labels_
    df['distance_point_to_cluster'] = KM.transform(matrix).min(axis=1)


def clustering(df, matrix):
    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    AP = AffinityPropagation()
    AP.fit(matrix)
    df['labels'] = AP.labels_


def printing_result(df):
    print(df['labels'].nunique())
    for label in df['labels'].value_counts().index[:10]:
        print('cluster {}'.format(label))
        tag = ''
        for index in df[df['labels'] == label].index:
            tag = tag + df.loc[index, 'heading'] + df.loc[index, 'content']
        print(google_df[google_df['labels'] == label]['distance_point_to_cluster'].mean())
        print(jieba.analyse.extract_tags(HanziConv.toSimplified(tag), topK=10))
        print(df['content'][df['labels'] == label])


def excel_output(df, name='default'):
    folder_path = 'output/'

    if not (os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    file_path = '{}{}{}'.format(folder_path, name, '.csv')
    df.to_csv(file_path)
    print('Export output to: {}'.format(file_path))


# code refactoring
# identify entity
# identify item connected to entity
# any systematic way to tune the cluster of result

google_df = read_from_file(list_url)
google_df = sim_to_trad(google_df)
google_df = cleaning(google_df)
google_df = drop_duplicates(google_df)
extract_tag(google_df)
google_tfidf = vectorizer(google_df, method='tfidf')
matrix = dimension_reduction(google_tfidf, method='nmf')
sil_score_df = KM_clustering_selection_sil_score(google_df, matrix)
KM_clustering(google_df, matrix, 186)
clustering(google_df, matrix)
printing_result(google_df)
google_df['cluster_size'] = google_df.groupby('labels').transform(pd.Series.count)['heading']
google_df = google_df.sort_values(by=['cluster_size', 'labels', 'distance_point_to_cluster'],
                                  ascending=[False, True, True])
excel_output(google_df)

#AP = AffinityPropagation()
#AP.fit(matrix)
#google_df['labels'] = AP.labels_
#sil_score_df.sort_values('silhouette_score')

sil_score_df.plot(x='number_cluster', y='silhouette_score')
