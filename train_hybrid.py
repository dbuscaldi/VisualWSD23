import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import clip
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from Models import *
import pickle

CAPTIONS=0
DIFFUSED=1
CLIPONLY=2
FULL=3

MODE=CAPTIONS

device = "cuda" if torch.cuda.is_available() else "cpu"

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
clip_model, preprocess = clip.load("ViT-B/32", device=device) #2 models in memory is too much

if MODE == CAPTIONS or MODE==FULL:
    print("loading training images captions embeddings...")
    ce=open("data/SBERT_capemb.pkl", "rb")
    cap_vecs=pickle.load(ce)
    ce.close()
    print("done.")

if MODE==DIFFUSED or MODE==FULL:
    print("loading CLIP diffused text embeddings...")
    dif=open("data/CLIP_diff_tr.pkl", "rb")
    difd=pickle.load(dif)
    dif.close()
    print("done.")

print("loading CLIP training img embeddings...")
cie=open("data/CLIP_train.pkl", "rb")
cie_dict=pickle.load(cie)
cie.close()
print("done.")

print("loading gold standard...")
gold=[]
with open("train_v1/train.gold.v1.txt") as g_file:
    for l in g_file:
        gold.append(l.strip())
print("done.")

print("preparing training data...")
train_data=[]
y_labels=[]
i=0
with open("train_v1/train.data.v1.txt") as t_file:
    for l in t_file:
        els=l.strip().split('\t')
        tgt_word=els[0]
        tgt_sen=els[1]
        images=els[2:]
        #SBERT embedding
        if MODE==DIFFUSED:
            t_emb = torch.squeeze(difd['train_img'+str(i)+'.jpg'])
        if MODE==CAPTIONS:
            t_emb = sbert_model.encode(tgt_sen, convert_to_numpy=False)
        if MODE==FULL:
            d_emb = torch.squeeze(difd['train_img'+str(i)+'.jpg'])
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
        scores=[]
        g_img=gold[i] #gold img
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for img in images:
            if g_img==img:
                scores.append((img, torch.tensor(1.0)))
            else:
                scores.append((img, torch.tensor(-1.0)))

        #scores.sort(key = lambda x: -x[1])
        for p in scores:
            if MODE == CAPTIONS or MODE == FULL: img_vec=cap_vecs.get(p[0], torch.zeros([384])).to(device)
            clip_vec=cie_dict.get(p[0], torch.zeros([512])).to(device)
            if MODE==DIFFUSED or MODE==CLIPONLY:
                img_emb=torch.squeeze(clip_vec) #only CLIP
            else:
                #text AND clip
                if MODE == CAPTIONS:
                    img_emb=torch.cat((img_vec, torch.squeeze(clip_vec)))
                if MODE == FULL:
                    #using CLIP to represent generation
                    img_emb=torch.cat((img_vec, torch.squeeze(clip_vec), torch.squeeze(clip_vec)))
            train_data.append((tgt_emb, img_emb))
            y_labels.append(p[1])

        i+=1
print("done.")

y_labels=torch.tensor(y_labels).to(device)

tr_dataset = list(zip(train_data, y_labels))
lengths = [int(len(tr_dataset)*0.9), int(len(tr_dataset)*0.1)]
train, valid = random_split(tr_dataset, lengths)

dataloader = DataLoader(train, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid, batch_size=16, shuffle=True)

size=(384+512)
if MODE==DIFFUSED or MODE==CLIPONLY: size=512
if MODE==FULL: size=384+512+512 #text+diffusion+CLIP <-> text+CLIP*2
network = TwinHybridNetwork(size).to(device)
optimizer = torch.optim.Adam(network.parameters())
criterion = nn.CosineEmbeddingLoss(margin=0.5)

num_epochs = 20

print("training model...")
torch.enable_grad()
last_avg_valid_loss=10.0
for epoch in range(num_epochs):
    tr_losses=[]
    for (input1, input2), label in dataloader:
        # Pass the inputs through the network
        (out1, out2) = network(input1.float(), input2.float())
        loss=criterion(out1,out2, label)
        tr_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print the average loss for the epoch
    valid_losses = []
    network.eval()
    #TODO: stop when validation loss increases (currently aroound epoch 5)
    for (v_1, v_2), label in valid_loader:
        (out1, out2) = network(v_1.float(), v_2.float())
        v_loss = criterion(out1, out2, label)
        valid_losses.append(v_loss.item())

    print(f'Epoch {epoch+1}, Training Loss: {np.mean(tr_losses)}')
    print(f'Epoch {epoch+1}, Validation Loss: {np.mean(valid_losses)}')
    if last_avg_valid_loss < np.mean(valid_losses):
        print("Validation loss is increasing, stopping training...")
        break
    last_avg_valid_loss=np.mean(valid_losses)

    network.train()

print("saving model...")
if MODE==CAPTIONS:
    model_filename='state_dict_hybN_m05_16b_tanh.pt'
if MODE==DIFFUSED:
    model_filename='state_dict_hybG_m08_16b.pt'
if MODE==CLIPONLY:
    model_filename='state_dict_hybC_m08_16b.pt'
if MODE==FULL:
    model_filename='state_dict_hybF_m08_16b.pt'

torch.save(network.state_dict(), model_filename)
print("finished!")
