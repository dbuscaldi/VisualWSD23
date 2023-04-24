import csv
import sys, re
import torch
import numpy as np

with open(sys.argv[1]) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        #print(row['feature'], row['feature_embedding'])
        s=row['feature_embedding']
        s=s.replace('[','')
        s=s.replace(']','').strip()
        s=re.sub('\s+', ',', s)
        vec=np.fromstring(s, dtype=float, sep=',')
        #print(vec)
        t=torch.tensor(vec)
        print(row['feature'], t.shape)
