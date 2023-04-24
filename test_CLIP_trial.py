import torch, sys, os
import numpy as np
import clip
from PIL import Image
import pickle

import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
"""
#this is if we use the captions and text_only embeddings. It seems captions are not very good
cap_vecs={}
with open("data/captions_trial.tsv") as cap_file:
    for l in cap_file:
        id, sen = l.strip().split('\t')
        s=clip.tokenize(sen).to(device)
        cap_vecs[id]=model.encode_text(s)
print("done.")
"""
print("loading img embeddings...")
cie=open("data/CLIP_trial.pkl", "rb")
#cie_dict=pickle.load(cie)
cie_dict=CPU_Unpickler(cie).load()
cie.close()

gold=[]
#testing on trial:
print("testing...")
with open("trial_v1/trial.gold.v1.txt") as g_file:
    for l in g_file:
        gold.append(l.strip())

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

test_data=[] #test data will have as head the sentence embedding and the rest will be the candidate ones
ty_labels=[] #1 for the right image, 0 for the others
i=0
MRRs=[]
with open("trial_v1/trial.data.v1.txt") as t_file:
    for l in t_file:
        els=l.strip().split('\t')
        tgt_word=els[0]
        tgt_sen=els[1]
        images=els[2:]
        g_img=gold[i]

        img_embs=[]
        for f in images:
            """
            #creating image embeddings on the fly
            fname=os.path.join("trial_v1/trial_images_v1/", f)
            image = preprocess(Image.open(fname)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            """
            #image_features=cap_vecs[f] #this is if we use only text encoding

            image_features = cie_dict[f]
            img_embs.append(image_features)
        text = clip.tokenize(tgt_sen).to(device)
        t_emb=model.encode_text(text)
        preds=[]
        j=0
        for e in img_embs:
            x=torch.squeeze(t_emb) #512 size
            y=torch.squeeze(e)
            preds.append((images[j],cos(x, y).item()))
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
