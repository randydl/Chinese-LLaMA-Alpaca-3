import os
import fire
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_example(row, choices, include_answer=True):
    prompt = row['Question']
    for choice in choices:
        prompt += f'\n{choice}. {row[choice]}'
    prompt += '\n答案：'
    if include_answer:
        prompt += f'{row["Answer"]}\n\n'
    return prompt


def gen_fewshot_prompt(df, n_shot):
    fewshot_examples = df.head(n_shot)
    choices = [col for col in df.columns if col not in ['Question', 'Answer']]
    prompt = '以下是中国关于半导体集成电路考试的单项选择题，请选出其中的正确答案。\n\n'
    for _, row in fewshot_examples.iterrows():
        prompt += format_example(row, choices)
    return prompt, choices


@torch.no_grad()
def evaluate_model(model, tokenizer, df, n_shot):
    accuracy_list = []
    results = []

    prompt_prefix, choices = gen_fewshot_prompt(df, n_shot)
    test_examples = df.iloc[n_shot:]

    for _, row in tqdm(test_examples.iterrows(), total=len(test_examples), leave=False):
        prompt = prompt_prefix + format_example(row, choices, include_answer=False)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        logits = model(input_ids).logits[:, -1].flatten()
        probs = logits[[tokenizer.encode(choice)[0] for choice in choices]].softmax(dim=0)
        predicted_choice = choices[probs.argmax(dim=0).item()]
        accuracy_list.append(predicted_choice == row['Answer'])
        row['Prediction'] = predicted_choice
        results.append(row)

    overall_accuracy = np.mean(accuracy_list)
    results_df = pd.DataFrame(results)
    return overall_accuracy, results_df


def main(model_path, data_dir, save_dir=None, n_shot=5, **kwargs):
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    save_dir = model_path / 'cxmt' if save_dir is None else Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation='sdpa'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    evaluation_results = {}
    csv_files = sorted(data_dir.glob('*.csv'))
    for csv_file in tqdm(csv_files):
        category = csv_file.stem
        csv_data = pd.read_csv(csv_file)
        csv_data = csv_data.loc[:, ~csv_data.columns.str.contains('^Unnamed')]
        accuracy, results_df = evaluate_model(model, tokenizer, csv_data, n_shot)
        results_df.to_csv(save_dir / csv_file.name, index=False)
        evaluation_results[category] = accuracy

    for category, accuracy in evaluation_results.items():
        print(f'Average accuracy {accuracy*100:.2f}% - {category}')


if __name__ == '__main__':
    fire.Fire(main)
