import os
import random
import shutil
path = "training_set"
files = os.listdir(path)
ss = []
for file in files:
    if "Annotation" not in file:
        pre = file.split(".",1)[0]
        ss.append(pre)

vals = random.sample(ss, 99)

trains= list(set(ss)-set(vals))

for val in vals:
    old1 = "training_set/" + val + ".png"
    old2 = "training_set/" + val + "_Annotation.png"
    old3 = "training_labels/" + val + "_Annotation_label.png"
    new1 = "val_set/" + val + ".png"
    new2 = "val_set/" + val + "_Annotation.png"
    new3 = "val_labels/" + val + "_Annotation_label.png"
    shutil.move(old1, new1)
    shutil.move(old2, new2)
    shutil.move(old3, new3)