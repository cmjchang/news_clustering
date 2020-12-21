# Google result clustering

Unsupervised clustering on google news result. 

# Usage

The code is Python 3 based. In order to run the code correctly, it is recommended to create a virtual environment and install the dependencies in ```requirements.txt``` as described below:

- Create a new python environment: `pyenv virtualenv 3.7 googleresult`
- Activate the google_result environement: `pyenv activate googleresult`
- Install the requirements: `pip install -r requirements.txt`
- Run the "python main.py" to generate required result.
- Run the "streamlit run app.py" for display

# Further details about the clustering

Current clustering methodology required manual selection of the number 
clusters (K-means) / some kind of hyperparameter (DBSCAN). This project 
is testing different combination of dimension reduction technology 
(PCA / NMF) and clustering method to create best result.

# Whole projects

This is a middle part of the entire projects.

The entire project is consist the following parts.

Webcrawling to generate news feed. (DONE)
Group similar news article together with clustering. (this part)
Classify different news article into pre-setted topics. (DONE)
Using varies NLP method to do sentiment analysis, NER.
Create knowledge based on the entity linkage.