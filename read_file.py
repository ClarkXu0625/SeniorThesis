import numpy as np
import tables
import tqdm
import os
import glob

	
DIR = "/media/clark/clark_drive1/data_file/AM11_4Tastes_191030_114043_unsorted"
file_name = "AM11_4Tastes_191030_114043_unsorted.h5"
atom = tables.IntAtom()
# Read EMG data from amplifier channels
hf5 = tables.open_file(os.path.join(DIR, file_name), 'r+')

hf5.close



pattern = os.path.join(DIR, "amp*")
files = glob.glob(pattern)
file_names = [os.path.basename(file) for file in files]


data = np.fromfile(os.path.join(DIR, file_names[0]), dtype = np.dtype('int16'))
print(data)