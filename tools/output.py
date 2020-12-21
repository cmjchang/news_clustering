import os

import jieba
from hanziconv import HanziConv


def printing_result(df):
    print(df['labels'].nunique())
    for label in df['labels'].value_counts().index[:10]:
        print('cluster {}'.format(label))
        tag = ''
        for index in df[df['labels'] == label].index:
            tag = tag + df.loc[index, 'heading'] + df.loc[index, 'content']
        print(df[df['labels'] == label]['distance_point_to_cluster'].mean())
        print(jieba.analyse.extract_tags(HanziConv.toSimplified(tag), topK=10))
        print(df['content'][df['labels'] == label])


def excel_output(df, name='default'):
    folder_path = 'output/'

    if not (os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    file_path = '{}{}{}'.format(folder_path, name, '.csv')
    df.to_csv(file_path)
    print('Export output to: {}'.format(file_path))
