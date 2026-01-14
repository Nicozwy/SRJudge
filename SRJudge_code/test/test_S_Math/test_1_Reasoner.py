import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
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

from vllm import LLM, SamplingParams

print("PyTorch version:", torch.__version__)


llm_model_pth = '../GRPO-main/GRPO_Reasoner'

MAX_NUM_SEQS = 32
MAX_MODEL_LEN = 8192*3//2

llm = LLM(
    model=llm_model_pth,
    max_num_seqs=MAX_NUM_SEQS,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,
    seed=2024,
)
tokenizer = llm.get_tokenizer()


dataset = load_dataset("csv", data_files={"test": "../result/test_predictions.csv"})

label_list = pd.read_csv("../result/test_predictions.csv")["label_text"].astype(str).unique().tolist()

print("标签列表:", label_list)

system_prompt = """你是一个数学习题知识点标注助手，请你根据我提供的'数学题目与解析'，在给出的5个候选知识点列表中挑选出最符合该题目的一个知识点,并将选择原因放入<reason>与</reason>之中，将答案放入\\boxed{}中"""

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"数学题目与解析为：{example['text']}，\n\n请你从{example['top5_predicted_labels_text']}中选出最符合的一个知识点，并将选择原因放入<reason>与</reason>之中，将答案放入\\boxed{{}}中。注意：无须解题，只需要挑选出最符合该数学题目与解析的知识点，并将原因放入<reason>与</reason>之中，将答案放入\\boxed{{}}中！"
            }
        ]
    }

dataset = dataset.map(make_conversation)

for split in dataset:
    if "label_text" in dataset[split].column_names:
        dataset[split] = dataset[split].rename_column("label_text", "solution")
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
    
def extract_reason_from_model_output(text):
    matches = re.findall(r"<reason>(.*?)</reason>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None
    
def convert_prompt(chat_prompt):
    return tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.90,
    max_tokens=1024*8,
    stop_token_ids=[tokenizer.eos_token_id],
)


y_true, y_pred, all_predictions,all_reasons = [], [], [],[]

test_examples = dataset["test"]

prompts = [convert_prompt(example["prompt"]) for example in test_examples]
print("开始批量生成...")

outputs = llm.generate(prompts, sampling_params)


for example, output in zip(test_examples, outputs):
    output_text = output.outputs[0].text
    print(output_text)
    prediction = extract_answer_from_model_output(output_text)
    answer = extract_answer_from_dataset(example["solution"])
    text_content = example["question"]
    reason =  extract_reason_from_model_output(output_text)

    if prediction is not None:
        prediction = prediction.strip()
        answer = answer.strip()
        final_pred = prediction


        if prediction not in label_list:
            print(f"⚠️ 用的是第一个知识点，index: {example.get('index', '未知')}")
            top5 = ast.literal_eval(example['top5_predicted_labels_text'])
            rand_label = top5[0]
            answer = answer.strip()
            final_pred = rand_label


        y_pred.append(final_pred)
        y_true.append(answer)
        all_predictions.append(final_pred)
        all_reasons.append(reason)

    else:
        top5 = ast.literal_eval(example['top5_predicted_labels_text'])
        rand_label = top5[0]
        answer = answer.strip()
        final_pred = rand_label


        y_pred.append(final_pred)
        y_true.append(answer)
        all_predictions.append(final_pred)
        print(f"⚠️ 无法提取预测，使用Top-1标签，index: {example.get('index', '未知')}")


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

df = pd.read_csv("../../result/test_predictions.csv")
assert len(df) == len(all_predictions), "样本数量不一致，无法合并预测结果。"

df["top5model_predict_label"] = all_predictions
df["reason"] = all_reasons
df.to_csv("../../result/reslut.csv", index=False)
print("✅ 结果已保存至 result/reslut.csv")
