import argparse
import logging

import os
import sys
import random
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score

import keras
import tensorflow as tf
import keras.layers as L
from keras import layers
import keras.backend as K
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.losses import binary_crossentropy
from keras.utils import pad_sequences
from keras.layers import Layer, Dropout, Dense, Input, Embedding, Bidirectional, LSTM, Concatenate


from src.configuration import load_config

logger = logging.getLogger(__name__)

# change as required
MODEL_CLASSES = {
    # "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    # "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    # "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    # "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    # "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}

def seed_everything(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=False,
        help="The input model directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--test_result_file",
        default="test_results.txt",
        type=str,
        required=False,
        help="The test_result",
    )

    parser.add_argument(
        "--test_prediction_file",
        default="test_predictions.txt",
        type=str,
        required=False,
        help="The test_result",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_finetune", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")

    # parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    # parser.add_argument(
    #     "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    # )
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    # parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument(
    #     "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    # )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    args = parser.parse_args()

if __name__ == "__main__":
    main()