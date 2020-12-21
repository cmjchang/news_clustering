import pandas as pd
import streamlit as st
from collections import Counter

df = pd.read_csv('output/default.csv')

df.to_html()

tag = Counter()
for cell in df['extract_tag']:
    tag.update(Counter(cell.split()))

message = []
for text, value in tag.most_common(10):
    message.append(' '.join([text, str(value)]))
str(tag.most_common(10))

print(message)

value = st.sidebar.slider('number of cluster to be displayed',
                          min_value=0,
                          max_value=int(df['labels'].max()),
                          value=(2, 5),
                          step=1,)

#df.groupby('labels')['distance_point_to_cluster'].mean().sort_values()


for label in df['labels'].value_counts().index[value[0]:value[1]]:
    df_subset = df[df['labels'] == label]
    st.write(f'cluster {label}, size {df_subset.shape[0]}')
    tag = Counter()
    for cell in df_subset['extract_tag']:
        tag.update(Counter(cell.split()))
    st.write(str(tag.most_common(10)))
    html_table = df_subset[['heading', 'content', 'extract_tag']].to_html(escape=False)
    st.write(html_table, unsafe_allow_html=True)
