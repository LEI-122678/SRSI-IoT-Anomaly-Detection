import torch
import pandas as pd
import sklearn

print("PyTorch:", torch.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)

print("CUDA disponível:", torch.cuda.is_available())

x = torch.randn(3, 3)
print("Tensor de teste:")
print(x)