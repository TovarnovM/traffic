import time
print('waiting for 5 seconds') 
time.sleep(5)

import torch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
