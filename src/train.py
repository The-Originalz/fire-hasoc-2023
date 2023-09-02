import os
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import Dataset
from transformers import create_optimizer
from sklearn.metrics import classification_report

from hasoc_models import create_model
from hasoc_preprocess import clean_df, get_collator
from logger import config, logger
from hasoc_tokenizers import get_tokenizer, tokenize_text

_SEED = 2023
_LANG = 'gujju'
_LABEL2ID = config['data']['labelmaps']['label2id']
_ID2LABEL = config['data']['labelmaps']['id2label']
_MODEL_NAME = "l3cube-pune/gujarati-bert"
_NUM_EPOCHS = 1
_BATCH_SIZE = 8

if __name__ == '__main__':

    logger.info(f'Reading data for {_LANG}...')
    data = pd.read_csv(os.path.join(config['data']['inp'], f'{_LANG}/train.csv'))
    logger.info(f'Training data loaded successfully, Length: {len(data)}')
    logger.info(f'Columns in Training data: {data.columns.to_list()}')

    test_data = pd.read_csv(os.path.join(config['data']['inp'], f'{_LANG}/test.csv'))
    logger.info(f'Test data loaded successfully, Length: {len(test_data)}')
    logger.info(f'Columns in Test data: {test_data.columns.to_list()}')
    logger.info(f'Label2Id: {_LABEL2ID}, Id2Label: {_ID2LABEL}')

    # # Cleaning and Preparing Test Data
    test_data = clean_df(test_data, _LABEL2ID, isTrain=False)

    tokenizer = get_tokenizer(_MODEL_NAME)
    tokenizer_wrapper = lambda x: tokenize_text(x, tokenizer)
    data_collator = get_collator(tokenizer)

    test = Dataset.from_pandas(test_data)
    test_tokenized = test.map(tokenizer_wrapper, batched=True)

    fold_path = os.path.join(config['data']['inp'], f'{_LANG}/folds/{_SEED}')
    dirs = os.listdir(fold_path)

    oof_preds = np.zeros((data.shape[0],))
    test_preds = np.zeros((test_data.shape[0], 2))

    for dir_name in dirs:
        dir_path = os.path.join(fold_path, dir_name)

        # Defining the Train and Val paths 
        train_df = pd.read_csv(os.path.join(dir_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(dir_path, 'val.csv'))
        
        # Cleaning and Prepareing the Data
        train_clean = clean_df(train_df, _LABEL2ID)
        val_clean = clean_df(val_df, _LABEL2ID)
        
        # Converting to HuggingFace Datasets
        train_ds = Dataset.from_pandas(train_clean)
        val_ds = Dataset.from_pandas(val_clean)
        
        # Tokenize the Data    
        train_tokenized = train_ds.map(tokenizer_wrapper, batched=True)
        val_tokenized = val_ds.map(tokenizer_wrapper, batched=True)
        
        # Defining the Parameters for Training
        batches_per_epoch = len(train_tokenized) // _BATCH_SIZE
        total_train_steps = int(batches_per_epoch * _NUM_EPOCHS)
        optimizer, schedule = create_optimizer(
            init_lr=2e-5,
            num_warmup_steps=0,
            num_train_steps=total_train_steps)
        
        # Define the Model 
        model = create_model(_MODEL_NAME, optimizer, _LABEL2ID, _ID2LABEL)

        # Converting to Tf Dataset for training
        train_set = model.prepare_tf_dataset(
            train_tokenized,
            shuffle=True,
            batch_size=_BATCH_SIZE,
            collate_fn=data_collator,
        )

        validation_set = model.prepare_tf_dataset(
            val_tokenized,
            shuffle=False,
            batch_size=_BATCH_SIZE,
            collate_fn=data_collator,
        )
        
        test_set = model.prepare_tf_dataset(
            test_tokenized,
            shuffle=False,
            batch_size=_BATCH_SIZE,
            collate_fn=data_collator,
        )
        
        # Define Model
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=3,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=False,
            start_from_epoch=0
        )
        
        history = model.fit(
            x=train_set,
            validation_data=validation_set,
            epochs=_NUM_EPOCHS,
            callbacks=[es])

        logits = model.predict(validation_set).logits
        y_preds = tf.argmax(tf.nn.sigmoid(logits), axis=1).numpy()
        oof_preds[val_df['index'].values] += y_preds

        # predict on test
        test_logits = model.predict(test_set).logits
        test_y_preds = tf.nn.sigmoid(test_logits)
        test_preds += test_y_preds/5
        

    y_true = data['label'].map(_LABEL2ID)
    print(classification_report(y_true, oof_preds))

    y_preds_test = tf.argmax(test_preds, axis=1).numpy()
    test_data['label'] = y_preds_test
    test_data['label'] = test_data['label'].map(_ID2LABEL)
    test_data['label'].value_counts()

    # test_data['id'] = test_id
    # test_data[['id', 'label']].head()

    # test_data[['id', 'label']].to_csv('./../out/sinhala_bert_baseline.csv', index=False)