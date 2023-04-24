import torch, sys
import torch.nn as nn
import numpy as np
import clip, pickle
from PIL import Image

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

IT_EN=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

if IT_EN:
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")

size=384 #SBERT size

cap_vecs={}
print("loading captions embeddings...")
with open("data/captions_test_small.tsv") as cap_file:
    for l in cap_file:
        id, sen = l.strip().split('\t')
        embedding = sbert_model.encode(sen, convert_to_numpy=False)
        cap_vecs[id]=embedding
print("done.")

gold=[]
print("testing...")
with open("test_data/test.data.v1.1.gold/en.test.gold.v1.1.txt") as g_file:
    for l in g_file:
        gold.append(l.strip())

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

test_data=[] #test data will have as head the sentence embedding and the rest will be the candidate ones
ty_labels=[] #1 for the right image, 0 for the others
i=0
MRRs=[]
with open("test_data/en.test.data.txt") as t_file:
    for l in t_file:
        els=l.strip().split('\t')
        tgt_word=els[0]
        tgt_sen=els[1]
        images=els[2:]
        tgt_emb = sbert_model.encode(tgt_sen, convert_to_numpy=False)

        candidate_embs=[]
        g_img=gold[i] #gold img
        labels=[]
        for img in images:
            if g_img==img:
                labels.append(torch.tensor(1.0))
            else:
                labels.append(torch.tensor(0.0))

            img_emb=cap_vecs.get(img, torch.zeros([384])).to(device)
            candidate_embs.append(img_emb.to(device))

        preds=[]
        j=0
        for c in candidate_embs:
            preds.append((images[j],cos(tgt_emb, c)))
            j+=1
        preds.sort(key = lambda x: -x[1])
        j=1
        #print(tgt_sen, g_img)
        #print(preds)
        for (c_img, score) in preds:
            if c_img == g_img:
                MRRs.append(1.0/float(j))
            j+=1
        i+=1
print('MRR per input:', MRRs)
print('average MRR:', np.mean(MRRs))
