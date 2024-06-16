import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from scripts.dataset import Flickr8kDataset,Only_images_Flickr8kDataset
from itertools import islice
from ldm.util import instantiate_from_config
from PIL import Image
import PIL
import torch
import numpy as np
import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm
import lpips as lp
from einops import rearrange, repeat
from torch import autocast
from tqdm import tqdm, trange
from transformers import pipeline
from scripts.qam import qam16ModulationTensor, qam16ModulationString
import time
from PIL import Image
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from SSIM_PIL import compare_ssim
from torchvision.transforms import Resize, ToTensor, Normalize, Compose


'''

INIT DATASET AND DATALOADER

'''
capt_file_path  =  "path/to/captions.txt"          #"G:/Giordano/Flickr8kDataset/captions.txt"
images_dir_path =  "path/to/Images"                #"G:/Giordano/Flickr8kDataset/Images/"
batch_size      =  1    

dataset = Only_images_Flickr8kDataset(images_dir_path)

test_dataloader=DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True)


'''
MODEL CHECKPOINT

'''


model_ckpt_path = "path/to/model-checkpoint" #"G:/Giordano/stablediffusion/checkpoints/v1-5-pruned.ckpt"  #v2-1_512-ema-pruned.ckpt"        
config_path     = "path/to/model-config"     #"G:/Giordano/stablediffusion/configs/stable-diffusion/v1-inference.yaml"



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = (512,512)#image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


    

def test(dataloader,
         snr=10,
         num_images=100,
         sampling_steps = 50,
         outpath="outpath"
         ):

    blip = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    transform = Compose([Resize((512,512), antialias=True), transforms.PILToTensor() ])

    lpips = lp.LPIPS(net='alex')


    sample_path = os.path.join(outpath,f'Test-TEXTONLY-sample-{snr}-{sampling_steps}')
    

    os.makedirs(sample_path, exist_ok=True)

    sample_orig_path = os.path.join(outpath,f'Test-TEXTONLY-sample-orig-{snr}-{sampling_steps}')

    os.makedirs(sample_orig_path, exist_ok=True)

    text_path = os.path.join(outpath,f'Test-TEXTONLY-text-{snr}-{sampling_steps}')

    os.makedirs(text_path, exist_ok=True)


    lpips_values = []
    time_values = []
    ssim_values = []

    i=0


    for batch in tqdm(dataloader,total=num_images):

            img_file_path = batch[0]

            #Open Image
            init_image = Image.open(img_file_path)

            #Automatically extract caption using BLIP model
            prompt_blip = blip(init_image)[0]["generated_text"]

            #Save Caption for Clip metric computation
            f = open(os.path.join(text_path, f"{i}.txt"),"a")        
            f.write(prompt_blip)
            f.close()


            #Introduce noise in the text (aka. simulate noisy channel)
            prompt_corrupted =  qam16ModulationString(prompt_blip,snr)

            #Compute time to reconstruct image
            time_start = time.time()
            #Reconstruct image using noisy text caption
            image_generated = pipe(prompt_corrupted,num_inference_steps=sampling_steps).images[0]
            time_finish = time.time()

            time_elapsed = time_finish - time_start
            time_values.append(time_elapsed)

            #Save images for subsequent FID and CLIP Score computation
            image_generated.save(os.path.join(sample_path,f'{i}.png'))
            init_image.save(os.path.join(sample_orig_path,f'{i}.png'))

            #Compute SSIM
            init_image_copy = init_image.resize((512, 512), resample=PIL.Image.LANCZOS)
            ssim_values.append(compare_ssim(init_image_copy, image_generated))

            #Compute LPIPS
            image_generated = (transform(image_generated) / 255) *2 -1
            init_image = (transform(init_image) / 255 ) *2 - 1
            lp_score=lpips(init_image.cpu(),image_generated.cpu()).item()
            lpips_values.append(lp_score)

            i+=1
            if i==num_images:
              break

    print(f'mean lpips score: {sum(lpips_values)/len(lpips_values)}')

    print(f'mean ssim score: {sum(ssim_values)/len(ssim_values)}')

    print(f'mean time score: {sum(time_values)/len(time_values)}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    #START TESTING

    test(test_dataloader,snr=10,num_images=100,outpath=outpath)
    
    test(test_dataloader,snr=8.75,num_images=100,outpath=outpath)
    
    test(test_dataloader,snr=7.50,num_images=100,outpath=outpath)
    
    test(test_dataloader,snr=6.25,num_images=100,outpath=outpath)
    
    test(test_dataloader,snr=5,num_images=100,outpath=outpath)
    
    