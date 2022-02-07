import json
import os

with open('./config.json') as f:
    d = json.load(f)

with open('./config.json') as f:
    d = json.load(f)

TARGET_FOLDERS = d['TARGET_FOLDERS']
IMPOSTER_FOLDERS = d['IMPOSTER_FOLDERS']
BATCH_SIZE = d["BATCH_SIZE"]
IMG_H = d["IMG_H"]
IMG_W = d["IMG_W"]
AUG_TYPE = d["AUG_TYPE"]
MODEL = d['MODEL']

WEIGHTS = d['WEIGHTS']
F1_EPOCH = d['F1_EPOCH']
F1_LR = d['F1_LR']
LOG_DIR = d['LOG_DIR']

print(50 * '*', 'CONFIG', 50 * '*')
print('TARGET_FOLDERS:')
for i in TARGET_FOLDERS:
    print(f'{i}, exists: {os.path.exists(i)}')

print('IMPOSTER_FOLDERS:')
for i in IMPOSTER_FOLDERS:
    print(f'{i}, exists: {os.path.exists(i)}')

print(f'BATCH_SIZE={BATCH_SIZE}')
print(f'image size={IMG_H}*{IMG_W}')
print(f'fine tune {MODEL} with pretrain {WEIGHTS}')
print(f'train {F1_EPOCH} with learning_rate{F1_LR}')
print(f'LOG_DIR: {LOG_DIR}, exists: {os.path.exists(LOG_DIR)}')
