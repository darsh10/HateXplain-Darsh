from tqdm import tqdm
import jsonlines
import random
random.seed(42)

def create_pairwise_data(file_name):

    reader = jsonlines.open(file_name,"r")
    sentences = []
    labels    = set()
    writer    = jsonlines.open("pair/"+file_name,"w")
    lists  = []
    for r in tqdm(reader):
        targets = r['target']
        #lists.append(r)
        if targets == ['None']:
            continue
        lists.append(r)
        for target in targets:
            r1 = dict(r)
            r1['gold_label'] = '1'
            r1['tar'] = target
            writer.write(r1)
            labels.add(target)
    for r in tqdm(lists):
        targets = r['target']
        new_targets = [x for x in labels if x not in targets]
        random.shuffle(new_targets)
        for target in new_targets[:len(targets)]:
            r1 = dict(r)
            r1['gold_label'] = '0'
            r1['tar'] = target
            writer.write(r1)
    print(labels)

create_pairwise_data("val.jsonl")
create_pairwise_data("train.jsonl")
