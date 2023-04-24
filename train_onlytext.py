import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import clip
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from Models import *


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2')

print("loading captions embeddings...")
cap_vecs={}
with open("data/captions.tsv") as cap_file:
    for l in cap_file:
        id, sen = l.strip().split('\t')
        embedding = model.encode(sen)
        cap_vecs[id]=torch.from_numpy(embedding)
print("done.")

gold=[]
with open("train_v1/train.gold.v1.txt") as g_file:
    for l in g_file:
        gold.append(l.strip())

train_data=[]
y_labels=[]
i=0
with open("train_v1/train.data.v1.txt") as t_file:
    for l in t_file:
        els=l.strip().split('\t')
        tgt_word=els[0]
        tgt_sen=els[1]
        images=els[2:]
        embedding = torch.from_numpy(model.encode(tgt_sen))
        scores=[]
        g_img=gold[i] #gold img
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for img in images:
            try:
                img_vec=cap_vecs[img]
            except KeyError:
                img_vec=torch.zeros([384])
            if g_img==img:
                scores.append((img, torch.tensor(1.0)))
            else:
                scores.append((img, cos(embedding, img_vec)))
        try:
            scores.sort(key = lambda x: -x[1])
            for p in scores:
                try:
                    img_vec=cap_vecs[p[0]]
                except KeyError:
                    img_vec=torch.zeros([384])

                train_data.append((embedding, img_vec))
                y_labels.append(p[1])
        except:
            print(scores)
        i+=1

#binarize y_labels to use CosineEmbeddingLoss
y_labels=[torch.tensor(-1.0) if el < 1.0 else el for el in y_labels]


tr_dataset = list(zip(train_data, y_labels))
lengths = [int(len(tr_dataset)*0.9), int(len(tr_dataset)*0.1)]
train, valid = random_split(tr_dataset, lengths)

dataloader = DataLoader(train, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid, batch_size=16, shuffle=True)

network = TwinNetwork().to(device)
optimizer = torch.optim.Adam(network.parameters())
criterion = nn.CosineEmbeddingLoss(margin=0.8)

num_epochs = 10 #in any case we'll stop after 10 epochs

for epoch in range(num_epochs):
    tr_losses=[]
    last_avg_valid_loss=10.0
    for (input1, input2), label in dataloader:
        # Pass the inputs through the network
        (out1, out2) = network(input1, input2)
        loss=criterion(out1,out2, label)
        tr_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print the average loss for the epoch
    valid_losses = []
    network.eval()
    for (v_1, v_2), label in valid_loader:
        (out1, out2) = network(v_1, v_2)
        v_loss = criterion(out1, out2, label)
        valid_losses.append(v_loss.item())

    print(f'Epoch {epoch+1}, Training Loss: {np.mean(tr_losses)}')
    print(f'Epoch {epoch+1}, Validation Loss: {np.mean(valid_losses)}')
    #stop when validation loss increases
    if last_avg_valid_loss < np.mean(valid_losses):
        print("Validation loss is increasing, stopping training...")
        break
    last_avg_valid_loss=np.mean(valid_losses)

    network.train()

model_filename='state_dict_2L_100_lReLU_m08_16b.pt'
torch.save(network.state_dict(), model_filename)
