from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


tokenizer = AutoTokenizer.from_pretrained(r"./SLM_Selector")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_val_labels = pd.read_csv('../../data/train.csv')['kn_id'].tolist() + pd.read_csv('../../data/val.csv')['kn_id'].tolist()
label_encoder = LabelEncoder()
label_encoder.fit(train_val_labels)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item


test_df = pd.read_csv('../../data/test.csv')
test_texts = test_df['content'].tolist()
test_labels_raw = test_df['kn_id'].tolist()
test_labels = label_encoder.transform(test_labels_raw)

test_dataset = TextDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


model_path = r'./SLM_Selector'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.cuda()


all_preds = []
all_labels = []
top5_correct = 0
total_samples = 0
top5_label_texts = []

with torch.no_grad():
    for batch in test_loader:
        labels = batch['labels']
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits

        # Top-1
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Top-5
        top5_preds = torch.topk(logits, k=5, dim=-1).indices  # (batch_size, 5)
        for i in range(labels.size(0)):
            pred_ids = top5_preds[i].cpu().tolist()
            pred_labels = label_encoder.inverse_transform(pred_ids)
            top5_label_texts.append(','.join([str(label) for label in pred_labels]))

            if labels[i].item() in pred_ids:
                top5_correct += 1
            total_samples += 1

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
top5_accuracy = top5_correct / total_samples

# 7. 计算指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)


print(f"测试集准确率(Accuracy): {accuracy:.4f}")
print(f"测试集精确率(Precision, macro): {precision:.4f}")
print(f"测试集召回率(Recall, macro): {recall:.4f}")
print(f"测试集F1分数(F1, macro): {f1:.4f}")
# print(f"测试集F1分数(F1, weighted): {f1:.4f}")


print(f"Top-5 覆盖率 (Top-5 Accuracy): {top5_accuracy:.4f}")


report = classification_report(
    all_labels,
    all_preds,
    target_names=[str(label) for label in label_encoder.classes_],
    zero_division=0
)
print(report)


test_df['predicted_label'] = label_encoder.inverse_transform(all_preds)
test_df['top10_predicted_labels'] = top5_label_texts
test_df.to_csv('../result/test_predictions.csv', index=False)

import pandas as pd
import ast


df = pd.read_csv("../result/test_predictions.csv")
label_df = pd.read_csv("../data/diaa_2_training_data_label.csv")

# id -> label
id_to_label = dict(zip(label_df["id"].astype(str), label_df["label"]))

label_to_id = {label: id_str for id_str, label in id_to_label.items()}


def map_label_to_id(label):
    return label_to_id.get(label, f"[未知:{label}]")


def map_ids_to_labels(candidate_str):
    try:
        ids = ast.literal_eval(candidate_str)
        label_list = [id_to_label.get(str(i), f"[未知:{i}]") for i in ids]
        return str(label_list)
    except Exception as e:
        return "[]"

def map_id_to_label(single_id):
    return id_to_label.get(str(single_id), f"[未知:{single_id}]")

df["chinese_bert_predicted_label_text"] = df["predicted_label"].apply(map_id_to_label)
df["top5_predicted_labels_text"] = df["top5_predicted_labels"].apply(map_ids_to_labels)
import ast
import random
#shuffle
# df["top5_predicted_labels_text"] = df["top5_predicted_labels_text"].apply(
#     lambda x: random.sample(ast.literal_eval(x), len(ast.literal_eval(x)))
# )
# print(df["predicted_label"].head())


df.to_csv("../result/test_predictions.csv", index=False)

import pandas as pd


df = pd.read_csv("../result/test_predictions.csv")

df.rename(columns={"label": "label_text", "content": "text"}, inplace=True)


columns_to_keep = [
    "top5_predicted_labels",
    "top5_predicted_labels_text",
    "text",
    "label_text",
    "index",
    "kn_id",
    "chinese_bert_predicted_label_text"
]
df_filtered = df[columns_to_keep]

df_filtered.to_csv("../result/test_predictions.csv", index=False)
