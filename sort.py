import os
import cv2
import glob
import shutil
from pathlib import Path

image_path = Path('/Users/sunghun/Downloads/conc')
out_path = Path('/Users/sunghun/Downloads/data')

images = glob.glob(f"{image_path}/image/*.jpg")
name_list = {}
for image in images:
    base_name = os.path.basename(image)
    real_name = base_name.split('_')[-1]
    name = real_name.split('-')[0]
    if name not in name_list.keys():
        name_list[name] = [base_name]
    else:
        name_list[name].append(base_name)
    # if name not in name_list.keys():
    #     name_list[name] = 1
    # else:
    #     name_list[name] += 1

cnt = 0
cnt_2 = 0

for i in name_list.keys():
    if len(name_list[i]) > 3 :
        shutil.copy(f"{image_path}/image/{name_list[i][0]}",f"{out_path}/val/conc/image/{name_list[i][0]}")
        shutil.copy(f"{image_path}/black/{name_list[i][0]}",f"{out_path}/val/conc/truth/{name_list[i][0]}")
        del name_list[i][0]
    if len(name_list[i]) > 20 :
        shutil.copy(f"{image_path}/image/{name_list[i][1]}", f"{out_path}/val/conc/image/{name_list[i][1]}")
        shutil.copy(f"{image_path}/black/{name_list[i][1]}", f"{out_path}/val/conc/truth/{name_list[i][1]}")
        name_list
        del name_list[i][1]

    for name in name_list[i]:
        shutil.copy(f"{image_path}/image/{name}", f"{out_path}/train/conc/image/{name}")
        shutil.copy(f"{image_path}/black/{name}", f"{out_path}/train/conc/truth/{name}")