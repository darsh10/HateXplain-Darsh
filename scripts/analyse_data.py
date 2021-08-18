import json
import pdb
import jsonlines
import numpy as np
import csv
from sentence_transformers import SentenceTransformer

sentence_embeddings = SentenceTransformer('paraphrase-distilroberta-base-v1')

def hate_analysis(split_name, data, splits):
    print("Split", split_name)
    reason_count = {}
    ctr = 0
    labels = set()
    sentences = set()
    writer = jsonlines.open(split_name+".jsonl","w")
    csv_writer = csv.writer(open(split_name+"_analysis.csv","w"))
    headings = ['Label','Sentence','Array']
    csv_writer.writerow(headings)
    #arrays = ['Rhetoric includes disagreeing at the idea/belief level. Responses include challenging claims, ideas, beliefs, or trying to challenge their view.','Rhetoric includes negative nonviolent actions associated with the group. Responses include nonviolent actions including metaphors.','Rhetoric includes nonviolent characterizations and insults.','Rhetoric includes subhuman and superhuman characteristics.','Rhetoric includes infliction of physical harm or death. Responses include calls for literal violence or metaphorical/aspirational physical harm or death.','Rhetoric includes literal killing by group. Responses include the literal death/elimination of a group.']
    arrays = ['false,incorrect,wrong,challenge,persuade,change minds','threatened,stole,outrageous act,poor treatment,alienate','rat,monkey,nazi,demon,cancer,monster','punched,raped,starved,torturing,mugging','killed,annhilate,destroy']
    representations = sentence_embeddings.encode(arrays)
    representations = representations/np.linalg.norm(representations,axis=1)[:,None]
    print(np.linalg.norm(representations,axis=1))
    all_sentences   = []
    all_labels      = []
    for d in data:
        if d in splits[split_name]:
            all_same = True
            current_label = data[d]['annotators'][0]['label']
            for annotation in data[d]['annotators']:
                if annotation['label'] != current_label:
                    all_same = False
                labels.add(annotation['label'])
                if annotation['label'] == 'hatespeech':
                    if ' '.join(data[d]['post_tokens']).strip() not in sentences:
                        sentences.add(' '.join(data[d]['post_tokens']).strip())
                    for t in annotation['target']:
                        reason_count[t] = reason_count.setdefault(t,0) + 1
            if all_same:
                dict = {'sentence':' '.join(data[d]['post_tokens']).strip(),\
                        'gold_label':current_label,'uid':0,'target':\
                        data[d]['annotators'][0]['target']}
                all_sentences.append(dict['sentence'])
                all_labels.append(dict['gold_label'])
                writer.write(dict)
    print(reason_count)
    print(ctr/3)
    print(labels)
    all_representations = sentence_embeddings.encode(all_sentences)
    all_representations = all_representations/np.linalg.norm(all_representations,axis=1)[:,None]
    all_dot_products    = np.argmax(np.dot(representations, np.transpose(all_representations)),axis=0)
    correct = 0
    total = 0
    for label,sentence,index in zip(all_labels,all_sentences,all_dot_products):
        csv_writer.writerow([label,sentence,arrays[index]])
        if label != 'normal':
            total += 1
            if 'hate' in label.lower():
                if index>2:
                    correct += 1
            else:
                if index<=2:
                    correct += 1
    closest_category    = np.argmax(all_dot_products,axis=0)
    print(correct/total)
    writer.close()

data = json.load(open("Data/dataset.json","r"))
splits=json.load(open("Data/post_id_divisions.json","r"))

hate_analysis("train",data,splits)
hate_analysis("val",data,splits)
hate_analysis("test",data,splits)
