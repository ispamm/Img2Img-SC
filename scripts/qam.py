import numpy as np
from scipy.special import erfc
import random
import torch
import math
import bitstring
import numpy as np

'''
    This python file is dedicated to the QAM Modulation and noisy channel estimation 
    
    we used K-ary discerete memoryless channel (DMC) where the crossover probability is given by 
    the bit error rate (BER) of the M-ary quadrature amplitude modulation (QAM). 
   
    For DMC, you can find the channel model from (9) in https://ieeexplore.ieee.org/abstract/document/10437659. 
    
    For the crossover probability, you assumed an AWGN channel where the BER is a Q-function 
    of the SNR and M: https://www.etti.unibw.de/labalive/experiment/qam/. 

'''

# Modulate Tensor in 16QAM transmission and noisy channel conditions
def qam16ModulationTensor(input_tensor,snr_db=10):

  message_shape = input_tensor.shape

  message = input_tensor

  #Convert tensor in bitstream
  bit_list = tensor2bin(message)

  #Introduce noise to the bitstream according to SNR
  bit_list_noisy = introduce_noise(bit_list,snr=snr_db)

  #Convert bitstream back to tensor 
  back_to_tensor = bin2tensor(bit_list_noisy)

  return back_to_tensor.reshape(message_shape)


# Modulate String in 16QAM transmission and noisy channel conditions
def qam16ModulationString(input_tensor,snr_db=10):

  message = input_tensor

  #Convert string to bitstream
  bit_list = list2bin(message)

  #Introduce noise to the bitstream according to SNR
  bit_list_noisy = introduce_noise(bit_list,snr=snr_db)

  #Convert bitstream back to list of char
  back_to_tensor = bin2list(bit_list_noisy)

  return "".join(back_to_tensor)





def introduce_noise(bit_list,snr=10,qam=16):

    # Compute ebno according to SNR 
    ebno = 10 ** (snr/10)

    # Estimate probability of bit error according to https://www.etti.unibw.de/labalive/experiment/qam/ 
    K = np.sqrt(qam)  # 4
    M = 2 ** K
    Pm = (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 / 2 / (M - 1) * K * ebno))
    Ps_qam = 1 - (1 - Pm) ** 2
    Pb_qam = Ps_qam / K

    bit_flipped = 0
    bit_tot = 0
    new_list = []
    for num in bit_list:
      num_new = []
      for b in num:
        
        if random.random() < Pb_qam:
          num_new.append(str(1 - int(b)))  # Flipping the bit
          bit_flipped+=1
        else:
          num_new.append(b)
        bit_tot+=1
      new_list.append(''.join(num_new))

    #print(bit_flipped/bit_tot)
    return new_list





def bin2float(b):
    ''' Convert binary string to a float.

    Attributes:
        :b: Binary string to transform.
    '''

    num = bitstring.BitArray(bin=b).float

    #print(num)
    if math.isnan(num) or math.isinf(num):
        
        num = np.random.randn()
      

    if num > 10:
     
      num=np.random.randn()

    if num < -10:
      
      num=np.random.randn()

    if num < 1e-2 and num>-1e-2:
      
      num = np.random.randn()

    return num


def float2bin(f):
    ''' Convert float to 64-bit binary string.

    Attributes:
        :f: Float number to transform.
    '''

    f1 = bitstring.BitArray(float=f, length=64)
    return f1.bin


def tensor2bin(tensor):

  tensor_flattened = tensor.view(-1).numpy()

  bit_list = []
  for number in tensor_flattened:
    bit_list.append(float2bin(number))

  
  return bit_list


def bin2tensor(input_list):
  tensor_reconstructed = [bin2float(bin) for bin in input_list]
  return torch.FloatTensor(tensor_reconstructed)


def string2int(char):
  return ord(char)


def int2bin(int_num):
  return '{0:08b}'.format(int_num)

def int2string(int_num):
  return chr(int_num)

def bin2int(bin_num):
  return int(bin_num, 2)


def list2bin(input_list):

  bit_list = []
  for number in input_list:
    bit_list.append(int2bin(string2int(number)))

  return bit_list

def bin2list(input_list):
  list_reconstructed = [int2string(bin2int(bin)) for bin in input_list]
  return list_reconstructed






