import torch, os
from diffusers import StableDiffusionPipeline, AltDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

LANG="en"

if LANG=="en":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to(device)
if LANG=="it":
    pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9", torch_dtype=torch.float16)
    pipe = pipe.to(device)
#skip=7885 #resume from line N

#remove NSFW filter to avoid black images
def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy

dir_path='data/generated_images/blacked'
to_redo=[]
for f in os.listdir(dir_path):
    id=f.replace('train_img','')
    id=id.strip('.jpg')
    to_redo.append(int(id))

i=0

with open("train_v1/train.data.v1.txt") as t_file:
    for l in t_file:
        if i not in to_redo :
            i+=1
            continue
        els=l.strip().split('\t')
        tgt_word=els[0]
        tgt_sen=els[1] #prompt
        if LANG=="en": image = pipe(f'a photo of {tgt_sen}').images[0]
        if LANG=="it": image = pipe(f'una foto di {tgt_sen}', num_inference_steps=25).images[0]
        image.save("data/generated_images/blacked/train_img"+str(i)+".jpg")
        i+=1
