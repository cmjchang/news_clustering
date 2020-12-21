from functools import reduce

import jieba
import jieba.analyse
from hanziconv import HanziConv


def extract_tag(df):
    for index in df.index:
        cut_heading = jieba.analyse.extract_tags(df.loc[index, 'heading'], topK=50)
        cut_content = jieba.analyse.extract_tags(df.loc[index, 'content'], topK=50)
        df.loc[index, 'extract_tag'] = reduce(lambda x, y: x + ' ' + y, cut_heading + cut_content)
        df.loc[index, 'len_cut_heading'] = len(cut_heading)
        df.loc[index, 'len_cut_content'] = len(cut_content)