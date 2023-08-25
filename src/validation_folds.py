import os
import random
import pandas as pd
from sklearn.model_selection import KFold
from logger import config, logger

_LANG = 'gujju'
_SEED = random.sample(config['hyps']['seeds'], 1)[0]
_KFOLD = True
_TARGET_COL = 'label'

if __name__ == '__main__':
    # read train data
    folder_path = os.path.join(config['data']['inp'], _LANG)
    file_path = os.path.join(folder_path, 'train.csv')

    train_df = None
    if os.path.exists(file_path):
        logger.info('Loading train file to create folds')
        train_df = pd.read_csv(file_path)
    else:
        logger.warning(f'Train file {file_path} is missing')

    label_counts = train_df['label'].value_counts().to_dict()
    logger.info(f'Label counts in training dataset: {label_counts}')
    
    # create folds subfolder to store the fixed splits
    folds_path = os.path.join(os.path.join(folder_path, f'folds/{_SEED}'))
    if not os.path.exists(folds_path):
        os.makedirs(folds_path)

    logger.info(f'Subfolder to save folds created at {folds_path}')

    if _KFOLD:
        kf = KFold(n_splits=config['hyps']['n_split'], shuffle=True, random_state=_SEED)

        logger.info(f'Creating {config["hyps"]["n_split"]} folds with {kf}')
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df[_TARGET_COL])):
            logger.info(f'Fold: {fold}')
            logger.info(f'Train samples: {len(train_idx)}')
            logger.info(f'Target distribution: {train_df.iloc[train_idx][_TARGET_COL].value_counts().to_dict()}')
            logger.info(f'Val Samples: {len(val_idx)}')
            logger.info(f'Target distribution: {train_df.iloc[val_idx][_TARGET_COL].value_counts().to_dict()}')
            
            sub_folder_path = os.path.join(folds_path, f'{_SEED}_{fold}')
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)

            train_df.iloc[train_idx].to_csv(os.path.join(sub_folder_path, 'train.csv'), index=False)
            train_df.iloc[val_idx].to_csv(os.path.join(sub_folder_path, 'val.csv'), index=False)
    pass