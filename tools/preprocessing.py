import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from hanziconv import HanziConv

from sklearn.decomposition import TruncatedSVD

def read_from_file(file_list):
    dfs = []
    for file in file_list:
        with open(file, 'rb') as f:
            soup = BeautifulSoup(f.read().decode('utf8'), features="html.parser")
        for articles in soup.find_all('div', class_='g'):
            if not re.search(re.compile('youtube\.com|img|books.google.com'), articles.prettify()):
                heading = articles.find('a').text
                content = articles.find('span', class_='st').text
                # replace date
                cleanr = re.compile(
                    u'[0-9]{1,2}[ ]?月[0-9]{1,2}[ ]?日|\.{3}|[.]{3}|\\xa0|On\.cc|i-Cable|China Press|\.\.')
                heading = re.sub(cleanr, '', heading)
                content = re.sub(cleanr, '', content)
                data = {"heading": heading, "content": content}
                data = pd.DataFrame(data, index=[0])
                dfs.append(data)
    google_df = pd.concat(dfs, ignore_index=True)
    return google_df


def sim_to_trad(series):
    series = series.apply(lambda x: HanziConv.toTraditional(x))
    return series

def trad_to_sim(series):
    series = series.apply(lambda x: HanziConv.toSimplified(x))
    return series


def cleaning(series):
    cleanr1 = re.compile(u'[0-9]{1,2}月[0-9]{1,2}日|\.{3}|[0-9] hours ago')
    recent_pattern = re.compile(u'[0-9]{1} day[s]? ago|[0-9]{1,2} hour[s]? ago|[0-9]{1,2} min[s]? ago')
    series = series.apply(lambda x: re.sub(cleanr1, '', x))
    series = series.apply(lambda x: re.sub(recent_pattern, 'recent', x))
    return series


def dimension_reduction(tfidf):
    source_dim = tfidf.shape[1]
    for best_n_components in np.logspace(np.log10(source_dim / 100), np.log10(source_dim / 1.5), 100).astype(int):
        pca = TruncatedSVD(n_components=best_n_components, random_state=42)
        pca.fit(tfidf)
        comp_check = pca.explained_variance_ratio_
        final_comp = best_n_components
        if comp_check.sum() > 0.85:
            break
    final_PCA = TruncatedSVD(n_components=final_comp, random_state=42)
    print(f"Using {final_comp} components, we can explain {comp_check}% of the variability in the original data.")
    return final_PCA.fit_transform(tfidf)