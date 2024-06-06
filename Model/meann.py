import pandas as pd
import numpy as np


# Load CSV files
df1 = pd.read_csv('results_focal.csv')

col_dice = df1['dice_coef']
col_recall = df1['recall']
col_precision = df1['precision']

print('focal')
print('dice:',np.mean(col_dice),'recall:',np.mean(col_recall),'precision:',np.mean(col_precision))




df1 = pd.read_csv('results_dice.csv')

col_dice = df1['dice_coef']
col_recall = df1['recall']
col_precision = df1['precision']
print('Dice')

print('dice:',np.mean(col_dice),'recall:',np.mean(col_recall),'precision:',np.mean(col_precision))





df1 = pd.read_csv('results_focaltrevesky.csv')

col_dice = df1['dice_coef']
col_recall = df1['recall']
col_precision = df1['precision']
print('results_focaltrevesky')

print('dice:',np.mean(col_dice),'recall:',np.mean(col_recall),'precision:',np.mean(col_precision))






df1 = pd.read_csv('results_trevesky.csv')

col_dice = df1['dice_coef']
col_recall = df1['recall']
col_precision = df1['precision']
print('results_trevesky')

print('dice:',np.mean(col_dice),'recall:',np.mean(col_recall),'precision:',np.mean(col_precision))

