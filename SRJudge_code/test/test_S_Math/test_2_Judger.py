import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
# os.environ["PYTHONMULTIPROCESSING_START_METHOD"] = "spawn"


import ast
import warnings
warnings.simplefilter('ignore')
import json
import time
import random
import re
import difflib
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from vllm import LLM, SamplingParams  #QWEN3 vllm==0.9.2    Deepseek and Qwen2.5 vllm == 0.7.2

print("PyTorch version:", torch.__version__)


llm_model_pth = '/data/pretrain_model/models/Qwen/Qwen3-32B-AWQ'

MAX_NUM_SEQS = 32
MAX_MODEL_LEN = 8192*3//2

llm = LLM(
    model=llm_model_pth,
    max_num_seqs=MAX_NUM_SEQS,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95,
    seed=2024,
)

tokenizer = llm.get_tokenizer()


dataset = load_dataset("csv", data_files={"test": "./reslut.csv"})

label_list = pd.read_csv("./reslut.csv")["label_text"].astype(str).unique().tolist()

print("标签列表:", label_list)

system_prompt = """你是一个数学习题知识点标注助手，无需解题，我希望你在下面的对话中,联系两个模型的预测，从“知识点列表”或“两个模型的预测结果”中帮助我挑选出最符合该题目的知识点，并将知识点放入\\boxed{}中！注意是知识点放入\\boxed{}中！！！"""

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"数学题目与解析为：{example['text']}，\n\n知识点列表为：{example['top5_predicted_labels_text']}\n\nchinese_bert_predicted_label：{example['chinese_bert_predicted_label_text']}"
                           f"\n\nQwen2.5-1.5B-Instruct-GRPO_predicted_label:{example['top5model_predict_label']} \n\nQwen2.5-1.5B-Instruct-GRPO_predicted_label预测的原因为{example['reason']}"
            }
        ]
    }



dataset = dataset.map(make_conversation)

for split in dataset:
    if "label_text" in dataset[split].column_names:
        dataset[split] = dataset[split].rename_column("label_text", "solution")
    # if "label_text" in dataset[split].column_names:
    #     dataset[split] = dataset[split].rename_column("label_text", "solution")
    if "text" in dataset[split].column_names:
        dataset[split] = dataset[split].rename_column("text", "question")

print("Sample prompt:", dataset["test"][0]["prompt"])


def extract_answer_from_dataset(text):
    if text is None:
        return None
    return text.strip().replace(',', '')

def extract_answer_from_model_output(text):
    matches = re.findall(r"oxed{(.*?)}", text)
    if matches:
        return matches[-1].strip()
    return None

def convert_prompt(chat_prompt):
    return tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    max_tokens=1024*8,
    stop_token_ids=[tokenizer.eos_token_id],
    # skip_special_tokens=True
)


y_true, y_pred, all_predictions = [], [], []

test_examples = dataset["test"]

prompts = [convert_prompt(example["prompt"]) for example in test_examples]
print("开始批量生成...")

outputs = llm.generate(prompts, sampling_params)
log_file = "prediction_log_Qwen3_32B_baseline.txt"
log_f = open(log_file, "w", encoding="utf-8")

for example, output in zip(test_examples, outputs):
    output_text = output.outputs[0].text
    prediction = extract_answer_from_model_output(output_text)
    answer = extract_answer_from_dataset(example["solution"])
    text_content = example["question"]
    chinese_bert_pred = example.get("chinese_bert_predicted_label_text", "N/A")
    top5_model_pred = example.get("top5model_predict_label", "N/A")

    if prediction is not None:
        prediction = prediction.strip()
        answer = answer.strip()
        final_pred = prediction

        if prediction not in label_list:
            print(f"⚠️ 用的是第一个知识点，index: {example.get('index', '未知')}")
            rand_label = top5_model_pred
            answer = answer.strip()
            final_pred = rand_label


        y_pred.append(final_pred)
        y_true.append(answer)
        all_predictions.append(final_pred)
    else:
        rand_label = top5_model_pred
        answer = answer.strip()
        final_pred = rand_label


        y_pred.append(final_pred)
        y_true.append(answer)
        all_predictions.append(final_pred)
        print(f"⚠️ 无法提取预测，使用Top-1标签，index: {example.get('index', '未知')}")


    log_f.write("========================================= 样本 ==================================================================\n")
    log_f.write(f"题目与解析:\n{text_content}\n")
    log_f.write("--------------------------\n")
    log_f.write(f"模型输出:\n{output_text.strip()}\n")
    log_f.write("--------------------------\n")
    log_f.write(f"提取的预测: {prediction}\n")
    log_f.write(f"最终预测标签: {final_pred}\n")
    log_f.write(f"真实标签: {answer}\n")
    log_f.write(f"Chinese BERT 预测: {chinese_bert_pred}\n")
    log_f.write(f"Qwen2.5 top5 预测: {top5_model_pred}\n")
    log_f.write("\n")


log_f.close()
print(f"✅ 预测日志已保存至 {log_file}")


print("=== Evaluation ===")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("=== Macro ===")
# Macro
print("Macro Precision:", precision_score(y_true, y_pred, average="macro", zero_division=0))
print("Macro Recall:", recall_score(y_true, y_pred, average="macro", zero_division=0))
print("Macro F1 Score:", f1_score(y_true, y_pred, average='macro'))

print("=== Weighted ===")
# Weighted
print("Weighted Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
print("Weighted Recall:", recall_score(y_true, y_pred, average="weighted", zero_division=0))
print("Weighted F1 Score:", f1_score(y_true, y_pred, average='weighted'))

df = pd.read_csv("./reslut.csv")
assert len(df) == len(all_predictions), "样本数量不一致，无法合并预测结果。"

df["finall_predict_label"] = all_predictions
df.to_csv("reslut_Qwen3_32B.csv", index=False)
print("✅ 结果已保存至 reslut_Qwen3_32B.csv")


def evaluate_and_log(y_true, y_pred, label, log_file):
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n========================================= {} 评估指标 =========================================\n".format(label))
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("=== Macro ===\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write("=== Weighted ===\n")
        f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
        f.write(f"Weighted Recall: {weighted_recall:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")

    print(f"✅ {label} 指标已记录到日志")

df = pd.read_csv("./reslut.csv")
df_result = pd.read_csv("reslut_Qwen3_32B.csv")
assert len(df) == len(df_result)

y_true = df["label_text"].astype(str).tolist()
bert_preds = df["chinese_bert_predicted_label_text"].astype(str).tolist()
top5_preds = df["top5model_predict_label"].astype(str).tolist()
final_preds = df_result["finall_predict_label"].astype(str).tolist()

evaluate_and_log(y_true, bert_preds, "Chinese BERT", log_file)
evaluate_and_log(y_true, top5_preds, "Qwen2.5-Top1", log_file)
evaluate_and_log(y_true, final_preds, "Final Prediction", log_file)

