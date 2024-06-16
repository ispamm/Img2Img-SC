import argparse
import scripts.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p_gt', '--path_gt', type=str,
                        default='G:\Giordano\stablediffusion\outputs\img2img-samples\samples-orig-10')
    parser.add_argument('-p', '--path', type=str,
                        default='G:\Giordano\stablediffusion\outputs\img2img-samples\samples-10') 
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*.png'.format(args.path_gt)))
    # real_names = list(glob.glob('{}/*.jpg'.format(args.path_gt)))
    print(real_names, args.path_gt)
    
    
    
    fake_names = list(glob.glob('{}/*.png'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_fid = 0.0
    fid_img_list_real, fid_img_list_fake = [],[]
    idx = 0
    for rname, fname in tqdm(zip(real_names, fake_names), total=len(real_names)):
        idx += 1
            
        

        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        fid_img_list_real.append(torch.from_numpy(hr_img).permute(2,0,1).unsqueeze(0))
        fid_img_list_fake.append(torch.from_numpy(sr_img).permute(2,0,1).unsqueeze(0))
        avg_psnr += psnr
        avg_ssim += ssim
        if idx % 10 == 0:
        # fid = Metrics.calculate_FID(torch.cat(fid_img_list_real,dim=0), torch.cat(fid_img_list_fake,dim=0))
        # fid_img_list_real, fid_img_list_fake = [],[]                        
        # avg_fid += fid
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))


    #last FID
    fid = Metrics.calculate_FID(torch.cat(fid_img_list_real,dim=0), torch.cat(fid_img_list_fake,dim=0))
    avg_fid += fid

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    # avg_fid = avg_fid / idx

    # log
    print('# Validation # PSNR: {}'.format(avg_psnr))
    print('# Validation # SSIM: {}'.format(avg_ssim))
    print('# Validation # FID: {}'.format(avg_fid))