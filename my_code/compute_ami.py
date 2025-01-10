from sklearn.metrics.cluster import adjusted_mutual_info_score
import json

pre = "_only_mt"
with open(f'val_set{pre}.json', 'r') as file:
    data = json.load(file)

with open(f'val_gt_labels{pre}.json', 'r') as file:
    gt = json.load(file)

pre = "_only_mt_baseline"
with open(f'val_messages{pre}.json', 'r') as file:
    messages = json.load(file)

def cut_eof(x):
    new_x = []
    for i in x:
        if i == 0:
            return new_x
        else:
            new_x.append(i)
    return new_x

data_labels = []
data_rel_class = []
for d in data:
    node_num = len(d["ids"])
    data_labels.extend(d["labels"][:node_num])
    data_rel_class.append(d["rel_class"])

concepts = data_rel_class

messages = [cut_eof(m) for m in messages]
all_m_ids = []
message_dict = {}
message_id = 0

concepts = concepts[:len(messages)]
print(concepts[:50])
print(messages[:50])

for m in messages:
    m_code = ';'.join([str(i) for i in m])
    if m_code not in message_dict:
        message_dict[m_code] = message_id
        message_id += 1

    all_m_ids.append(message_dict[m_code])
        

# print(all_m_ids[:50])
# print(len(gt), len(data_labels))
for i, d in enumerate(data_labels[:len(gt)]):
    # print(i)
    print(gt[i])
    if d != gt[i]: 
        print(i, d, gt[i])

# Compute the Adjusted Mutual Information
ami_score = adjusted_mutual_info_score(concepts, all_m_ids)

print(f"Adjusted Mutual Information (AMI) Score {pre}: {ami_score}")

