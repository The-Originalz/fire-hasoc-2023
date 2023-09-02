import re
import pandas as pd
from transformers import DataCollatorWithPadding

def clean_df(df: pd.DataFrame, label2id: dict, isTrain=True) -> pd.DataFrame:
    df = df.copy()

    # Removing @tags 
    pattern = r'@\w+'
    df["text"] = df["text"].apply(lambda x: re.sub(pattern, '', x))

    if isTrain:
        # Transforming Categorical Values to Numericals
        df["label"] = df["label"].apply(lambda x: label2id[x])
            
        return df.loc[:, ['text', 'label']]
        
    return df.loc[:, 'text'].to_frame().reset_index(drop=True)

def clean_tweet(tweet: str):
    pass

def get_collator(tokenizer):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')
    return collator
    pass