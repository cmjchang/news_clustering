import numpy as np
import glob

from tools.read_data import read_from_file
from tools.preprocessing import sim_to_trad, cleaning, dimension_reduction
from tools.extract_tag import extract_tag
from tools.vectorization import vectorizer
from tools.output import excel_output
from tools.clustering import gini

from dynamic_clustering.dynamic_clustering import dynamic_clustering
from sklearn.decomposition import TruncatedSVD

# import dataset.
list_url = glob.glob('CCCC/[0-9]?.html')
google_df = read_from_file(list_url)
# cleaning
google_df.loc[:, 'heading'] = sim_to_trad(google_df['heading'])
google_df.loc[:, 'content'] = sim_to_trad(google_df['content'])
google_df.loc[:, 'heading'] = cleaning(google_df['heading'])
google_df.loc[:, 'content'] = cleaning(google_df['content'])
google_df = google_df.drop_duplicates(['heading', 'content'])
# enrichment
extract_tag(google_df)
# vectorization
google_tfidf = vectorizer(google_df, method='tfidf')
matrix = dimension_reduction(google_tfidf)

# Elbow Method for K means
from sklearn.cluster import KMeans
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans(random_state=42)
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2, 30), timings=True)
visualizer.fit(matrix)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# clustering
cluster_structure = dynamic_clustering(matrix, delta=0.0006, verbose=False)
# visualization
cluster_idx = np.array(cluster_structure['idx'])
google_df['labels'] = cluster_idx

print('Number of clusters', len(cluster_structure['centers']))

v = np.array(cluster_structure['NSamples'])
bins, result, gini_val = gini(v)
excel_output(google_df)
