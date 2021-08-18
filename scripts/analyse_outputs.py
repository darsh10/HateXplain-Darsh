import jsonlines

reader1 = jsonlines.open("val.jsonl","r")
reader2 = jsonlines.open("/data/rsg/nlp/darsh/pytorch-pretrained-BERT/hate_detector_output/preds.jsonl","r")

correct_labels = {}
incorrect_labels = {}
for r1,r2 in zip(reader1,reader2):
    label = r2['correct']
    for target in r1['target']:
        if label:
            correct_labels[target] = correct_labels.setdefault(target,0) + 1
        else:
            incorrect_labels[target] = incorrect_labels.setdefault(target,0) + 1

accuracy_labels = {}
for label in correct_labels:
    accuracy_labels[label] = correct_labels[label]/(correct_labels[label]+incorrect_labels[label])

print(accuracy_labels)
