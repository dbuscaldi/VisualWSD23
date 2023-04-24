import sys, os
import torch
import clip

from PIL import Image

import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

f_dict={}
#dir_path='trial_v1/trial_images_v1' 'train_v1/train_images_v1' 'data/generated_trial' 'test_data/test_images_resized'
dir_path='data/generated_test_it'
for f in os.listdir(dir_path):
    fname=os.path.join(dir_path, f)
    try:
        with torch.no_grad():
            image = preprocess(Image.open(fname)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            f_dict[f]=image_features
    except: #avoid image decompression bomb error
        continue
    #print(image_features)

print("saving Img CLIP embeddings...")
out_f=open("CLIP_gen_test_it.pkl", "wb")
pickle.dump(f_dict, out_f)
out_f.close()
