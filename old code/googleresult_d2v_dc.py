import pandas as pd
import numpy as np
import glob

from tools.read_data import read_from_file
from tools.preprocessing import sim_to_trad, cleaning
from tools.extract_tag import extract_tag
from tools.vectorization import vectorizer, dimension_reduction
from tools.clustering import km_clustering, km_clustering_selection, km_clustering_selection_sil_score, clustering
from tools.output import printing_result, excel_output

from dynamic_clustering.dynamic_clustering import dynamic_clustering
from dynamic_clustering.visualize import show_seg
import matplotlib.pyplot as plt

# jieba_path = r'C:\Users\Jimmy\Anaconda3\Lib\site-packages\jieba'
# load jieba dictionary
# jieba.set_dictionary(os.path.join(jieba_path, 'dict.txt'))

def gini(v):
    bins = np.linspace(0., 100., 11)
    total = float(np.sum(v))
    yvals = []
    for b in bins:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    return bins, yvals, gini_val



list_url = glob.glob('CCCC/[0-9]?.html')
google_df = read_from_file(list_url)
google_df.loc[:, 'heading'] = sim_to_trad(google_df['heading'])
google_df.loc[:, 'content'] = sim_to_trad(google_df['content'])
google_df.loc[:, 'heading'] = cleaning(google_df['heading'])
google_df.loc[:, 'content'] = cleaning(google_df['content'])
google_df = google_df.drop_duplicates(['heading', 'content'])
extract_tag(google_df)
google_tfidf = vectorizer(google_df, method='tfidf')
matrix = dimension_reduction(google_tfidf, method='nmf', n_components=25)

cluster_structure = dynamic_clustering(matrix, delta=0.0006, verbose=False)
cluster_idx = np.array(cluster_structure['idx'])
google_df['labels'] = cluster_idx

print(len(cluster_structure['centers']))
## visualize segmentation
#show_seg(cluster_structure['idx'], n_clusters=len(cluster_structure['centers']))
#plt.show()

v = np.array(cluster_structure['NSamples'])
bins, result, gini_val = gini(v)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(bins, result, label="observed")
plt.plot(bins, bins, '--', label="perfect eq.")
plt.xlabel("fraction of population")
plt.ylabel("fraction of wealth")
plt.title("GINI: %.4f" %(gini_val))
plt.legend()
plt.subplot(2, 1, 2)
plt.hist(v, bins=20)

excel_output(google_df)
