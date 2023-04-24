import torch, sys
import torch.nn as nn
import numpy as np
from Models import *
import clip, pickle
from PIL import Image

from sentence_transformers import SentenceTransformer

CAPTIONS=0
DIFFUSED=1
CLIPONLY=2
FULL=3

MODE=DIFFUSED
LANG="it"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

size=(384+512)
if MODE==DIFFUSED or MODE==CLIPONLY: size=512
if MODE==FULL: size=384+1024

network = TwinHybridNetwork(size).to(device)
#model_filename='state_dict_2L_100_lReLU.pt'
model_filename=sys.argv[1]

print("loading state dict...")
network.load_state_dict(torch.load(model_filename))

cap_vecs={}
if MODE != DIFFUSED:
    print("loading captions embeddings...")
    with open("data/captions_test_small.tsv") as cap_file:
        for l in cap_file:
            id, sen = l.strip().split('\t')
            embedding = sbert_model.encode(sen, convert_to_numpy=False)
            cap_vecs[id]=embedding
    print("done.")

if MODE==DIFFUSED or MODE==FULL:
    print("loading CLIP diffused embeddings...")
    if LANG=="en":
        dif=open("data/CLIP_gen_test.pkl", "rb")
    else: #Italian
        dif=open("data/CLIP_gen_test_it.pkl", "rb")
    difd=pickle.load(dif)
    dif.close()
    print("done.")

print("loading candidate img embeddings...")
cie=open("data/CLIP_test.pkl", "rb")
cie_dict=pickle.load(cie)
cie.close()

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
network.eval()

test_data=[] #test data will have as head the sentence embedding and the rest will be the candidate ones

i=0
MRRs=[]
with open("test_data/"+LANG+".test.data.txt") as t_file:
    for l in t_file:
        els=l.strip().split('\t')
        tgt_word=els[0]
        tgt_sen=els[1]
        images=els[2:]
        if MODE==DIFFUSED:
            t_emb = torch.squeeze(difd['test_img'+str(i)+'.jpg'])
        if MODE==CAPTIONS:
            t_emb = sbert_model.encode(tgt_sen, convert_to_numpy=False)
        if MODE==FULL:
            d_emb = torch.squeeze(difd['test_img'+str(i)+'.jpg'])
            ct_emb = sbert_model.encode(tgt_sen, convert_to_numpy=False)
            t_emb=torch.cat((ct_emb, d_emb)) #representing query with text AND diffusion
        #CLIP embedding
        with torch.no_grad():
            text = clip.tokenize(tgt_sen).to(device)
            tc_emb=torch.squeeze(clip_model.encode_text(text))
        #concatentate
        tgt_emb=torch.cat((t_emb, tc_emb))
        if MODE==DIFFUSED:
            tgt_emb=t_emb
        if MODE==CLIPONLY:
            tgt_emb=tc_emb #only CLIP

        candidate_embs=[]
        for img in images:
            if MODE == CAPTIONS or MODE==FULL: img_vec=cap_vecs.get(img, torch.zeros([384])).to(device)
            clip_vec=cie_dict.get(img, torch.zeros([512])).to(device)
            if MODE==DIFFUSED or MODE==CLIPONLY:
                img_emb=torch.squeeze(clip_vec) #only CLIP
            else:
                if MODE==CAPTIONS:
                    img_emb=torch.cat((img_vec, torch.squeeze(clip_vec)))
                if MODE==FULL:
                    #using CLIP to represent generation
                    img_emb=torch.cat((img_vec, torch.squeeze(clip_vec), torch.squeeze(clip_vec)))
            candidate_embs.append(img_emb.to(device))

        preds=[]
        base_sim=[]
        j=0
        for c in candidate_embs:
            (out1, out2) = network(tgt_emb.float(), c.float())
            score=cos(out1, out2)
            preds.append((images[j],score.item()))
            base_sim.append(cos(tgt_emb, c))
            j+=1
        preds.sort(key = lambda x: -x[1])

        #print(preds)
        res=[]
        for (c_img, score) in preds:
            res.append(c_img)
        i+=1
        print('\t'.join(res))
