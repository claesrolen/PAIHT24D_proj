import os
from os import listdir
import annotate
import pandas as pd

data_dirs = [
    "envdata/indoor/APPLAUSE",
    "envdata/indoor/ARENA",
    "envdata/indoor/COFFE_SHOP",
    "envdata/indoor/OFFICE",
    "envdata/indoor/SCHOOL",
    "envdata/indoor/SHOPPING_MALL",
    "envdata/outdoor/BUSSTATION",
    "envdata/outdoor/FIREWORKS",
    "envdata/outdoor/HARBOR",         
    "envdata/outdoor/NATURE",
    "envdata/outdoor/SCHOOLYARD",
    "envdata/outdoor/STREET",
    "envdata/outdoor/SUBWAY",
]
 
out_dir = "./SPLIT_ENV"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


ix_path = out_dir + "/index.csv"
HAS_INDEX = False
if os.path.isfile(ix_path):
    # Get dataframe, supports up to 50 columns, remove  NaN columns
    df = pd.read_csv(ix_path, header=None, names=range(50))
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
