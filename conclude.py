import argparse
import os
from os.path import join
import pandas as pd

parser = argparse.ArgumentParser(description='hyper-parameter search')
parser.add_argument('--data_dir', type=str, default="out", help='data_dir')
parser.add_argument('--dataset', type=str, default="MUTAG", help='dataset')
args = parser.parse_args()

df_list = []
keys = []

for root, dirs, files in os.walk(join(args.data_dir)):
    for name in dirs:
        _name = name.split('-')
        if _name[1] == args.dataset:
            csv = join(args.data_dir, name, 'grid_search.csv')
            if os.path.isfile(csv):
                df_list.append(pd.read_csv(csv))
                keys.append(_name[3] + '_' + _name[5] + '_' + _name[7][0])
df = pd.concat(df_list, keys=keys)
df.to_csv(join(args.data_dir, 'conclude-'+args.dataset+'.csv'))
print('done')



