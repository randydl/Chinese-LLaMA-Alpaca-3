import os
import fire
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_example(df, idx, include_answer=True):
    df_row = df.loc[idx, 'Question':'Answer']
    prompt = df_row['Question']
    for choice in df_row.index[1:-1]:
        prompt += f'\n{choice}. {df_row[choice]}'
    prompt += '\n答案：'
    if include_answer:
        prompt += f'{df_row["Answer"]}\n\n'
    return prompt


def gen_prompt(df, fewshot):
    prompt = '以下是关于半导体集成电路考试的单项选择题，请选出其中的正确答案。\n\n'
    for idx in df[:fewshot].index:
        prompt += format_example(df, idx)
    return prompt


@torch.no_grad()
def model_eval(model, tokenizer, df, fewshot):
    score = []
    result = df[fewshot:].copy(deep=True)

    for idx in tqdm(result.index, leave=False):
        prompt = gen_prompt(df, fewshot)
        prompt += format_example(df, idx, False)

        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        logits = model(input_ids).logits[:, -1].flatten()
        choices = df.loc[idx, 'Question':'Answer'].index[1:-1]
        probs = logits[[tokenizer.encode(x)[-1] for x in choices]].softmax(dim=0)
        pred = choices[probs.argmax(dim=0).item()]
        label = df.loc[idx, 'Answer']
        score.append(pred == label)
        result.loc[idx, 'Prediction'] = pred

    acc = np.mean(score)
    return acc, result


def main(model_path, data_dir, save_dir=None, fewshot=5):
    model_path = Path(model_path)
    data_dir = Path(data_dir)

    save_dir = model_path/'cxmt' if save_dir is None else Path(save_dir)
    # save_dir.mkdir(parents=True, exist_ok=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    result = {}
    csv_files = sorted(data_dir.glob('*.csv'))
    for csv_path in tqdm(csv_files):
        category = csv_path.stem
        csv_info = pd.read_csv(csv_path)
        acc, res = model_eval(model, tokenizer, csv_info, fewshot)
        res.to_csv(save_dir/csv_path.name, index=False)
        result[category] = acc

    for k, v in result.items():
        print(f'Average accuracy {v*100:.2f} - {k}')


if __name__ == '__main__':
    main(
        '/home/app.e0016372/models/Meta-Llama-3-8B',
        '/home/app.e0016372/data/temp'
    )
