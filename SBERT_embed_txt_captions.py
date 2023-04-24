import sys, os
import torch
from sentence_transformers import SentenceTransformer

import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("processing...")
cap_vecs={}
with open("data/captions.tsv") as cap_file:
    for l in cap_file:
        id, sen = l.strip().split('\t')
        embedding = sbert_model.encode(sen, convert_to_numpy=False)
        cap_vecs[id]=embedding
print("done.")

print("saving SBERT caption embeddings...")
out_f=open("SBERT_capemb.pkl", "wb")
pickle.dump(cap_vecs, out_f)
out_f.close()
