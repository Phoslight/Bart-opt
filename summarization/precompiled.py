import gc
import math
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # work around with mps bug
os.environ["WANDB_DISABLED"] = "true"
import random
import logging
import json
from typing import Any, List, Optional
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
from datasets import Dataset
from transformers import BartTokenizerFast, BartConfig, BartForConditionalGeneration, pipeline
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.bart.modeling_bart import shift_tokens_right
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

model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizerFast.from_pretrained(model_name)  # type: BartTokenizerFast

def print_size(model: nn.Module):
    tmp_file = "/tmp/tmp.pth"
    torch.save(model.state_dict(), tmp_file)

    origin_size = os.path.getsize(tmp_file)
    print(f"Model size: {origin_size / (1024**3):.2f} GB")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
device_summary = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # work around with mps bug