from slice_dataset import SliceDataset
from augmenter import AugDataset
from base_model import BaseModels
from training import TrainModel

from config import *
from support import fin


def pipeline_wrapper():
    # step 1 - create DS
    slice_dataset = SliceDataset(target_folders=TARGET_FOLDERS,
                                 imposter_folders=IMPOSTER_FOLDERS,
                                 img_h=IMG_H,
                                 img_w=IMG_W,
                                 batch_size=BATCH_SIZE,
                                 )

    train_dataset = slice_dataset.create_datasets()

    # step 2 - add augmentation
    augmenter = AugDataset()
    aug_dataset = augmenter.set_aug(train_dataset, AUG_TYPE)

    # step 3 - build model
    nn_model = BaseModels(model_name=MODEL,
                          img_h=IMG_H,
                          img_w=IMG_W,
                          weights=WEIGHTS)

    model = nn_model.build_model()

    # step 4 - train
    trainer = TrainModel(logdir=LOG_DIR)
    trainer.create_subdirs()

    model = nn_model.full_training_state_1(model)
    trainer.train(model=model,
                  train_dataset=train_dataset,
                  base_learning_rate=F1_LR,
                  epochs=F1_EPOCH,
                  train_label='F1')

    fin()
