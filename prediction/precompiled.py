import gc
import math
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # work around with mps bug
os.environ["WANDB_DISABLED"] = "true"
import time
import random
import logging
import json
from itertools import islice
from typing import Any, List, Optional
from abc import ABC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch import Tensor, Size
from torch.utils.data import DataLoader
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Function
from datasets import Dataset, load_dataset
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, pipeline
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import Conv1D
from transformers.trainer_utils import get_last_checkpoint
import nltk
nltk.download("punkt", quiet=True)
import evaluate
rouge_score = evaluate.load("rouge")
from ipywidgets import widgets
from IPython.display import clear_output, display

import warnings
warnings.filterwarnings('ignore')
# Ignore "Some non-default generation parameters are set in the model config..." message.
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_PATH = "../input/"

def create_work_dir(dir: str):
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    return dir

model_name = 'gpt2'  # 548M

def print_size(model: nn.Module):
    tmp_file = "/tmp/tmp.pth"
    torch.save(model.state_dict(), tmp_file)

    origin_size = os.path.getsize(tmp_file)
    print(f"Model size: {origin_size / (1024**3):.2f} GB")

class ABCConv1D(Conv1D, ABC):
    def linear(self, x, weight):  # transposed
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        torch.mps.empty_cache()
    # time.sleep(2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
device_gen = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # work around with mps bug