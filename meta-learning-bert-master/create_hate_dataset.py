import json
import pdb
from tqdm import tqdm
import random
random.seed(42)

dataset = json.load(open("../Data/dataset.json","r"))
post_ids= json.load(open("../Data/post_id_divisions.json","r"))

data_points = []

for post in dataset:
    split = 'train'
    if post in post_ids['test']:
        split = 'test'
    if post in post_ids['val']:
        split = 'val'
    text = " ".join(dataset[post]['post_tokens']).strip()
    annotations = dataset[post]['annotators']
    annotation_labels = [x['label'] for x in annotations]
    annotation_targets= list(set(sum([x['target'] for x in annotations],[])))
    sorted(annotation_labels)
    target_label = annotation_labels[int(len(annotation_labels)//2 + 1)]
    target_domain= random.choice(annotation_targets)
    data_point   = {'text':text, 'split':split, 'label':target_label,\
            'domain':target_domain}
    data_points.append(data_point)

json.dump(data_points, open("hate_dataset.json","w"))
