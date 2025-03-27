import os
from os import listdir
import annotate
import pandas as pd

'''
data_dirs = [
    "gundata/AK-12",
    "gundata/AK-47",
    "gundata/IMI Desert Eagle",
    "gundata/M16",
    "gundata/M249",
    "gundata/M4",
    "gundata/MG-42",
    "gundata/MP5",
    "gundata/Zastava M92"
]
out_dir = "./SPLIT_GUNS"
'''
data_dirs = ["EXTRA_COMBO"]
out_dir = "./SPLIT_COMBO"

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


ix_path = out_dir + "/index.csv"
HAS_INDEX = False
if os.path.isfile(ix_path):
    # Get dataframe, supports up to 200 columns, remove  NaN columns
    df = pd.read_csv(ix_path, header=None, names=range(200))
    df.dropna(axis=1, how="all", inplace=True)
    HAS_INDEX = True

for in_dir in data_dirs:
    for fname in os.listdir(in_dir):
        if fname.endswith(".wav"):
            prefix = fname.split(".wav")[0]
            if HAS_INDEX and df[0].str.contains(prefix, regex=False).any() == True:
                print(f"Duplicate index: {prefix}")
            else:
                print(f"Processing: {fname}")
                ann = annotate.Annotate(in_dir + "/" + fname, out_dir,LEN=4096)
                ann.process()
