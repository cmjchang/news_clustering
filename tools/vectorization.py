from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def vectorizer(df, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(df['extract_tag'])

    if method == 'count':
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(df['extract_tag'])





def doc_2_vec(df, vector_size=25, window=5, min_count=1):
    common_texts = df['extract_tag'].str.split()
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(documents, vector_size=vector_size, window=window, min_count=min_count, workers=8)
    return model.docvecs.vectors_docs