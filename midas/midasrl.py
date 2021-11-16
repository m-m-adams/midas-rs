from midas_cores import CMSCounter, MidasR
import torch
a = torch.cuda.is_available()
n = torch.cuda.get_device_name(0)

print(a, n)
