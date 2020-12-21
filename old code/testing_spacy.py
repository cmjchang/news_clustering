import pandas as pd
import numpy as np
import glob

from spacy.lang.zh import Chinese
import spacy

from tools.read_data import read_from_file

# Load with "default" model provided by pkuseg
cfg = {"pkuseg_model": "default", "require_pkuseg": True}
nlp = Chinese(meta={"tokenizer": {"config": cfg}})

spacy.load('zh_core_web_sm')

list_url = glob.glob('CCCC/[0-9]?.html')
google_df = read_from_file(list_url)

text = '王小明在北京的清华大学读书'
# spacy NER
spacy_name_list = []
doc = nlp(str(text))
print([(w.text, w.pos_) for w in doc])
for entity in doc.ents:
    if entity.label_ == 'PERSON':
        spacy_name_list.append(entity.text)