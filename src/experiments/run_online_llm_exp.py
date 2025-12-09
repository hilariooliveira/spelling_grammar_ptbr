import os
import json
import sys
import time
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from exp_utils import build_binary_corpus, build_multi_class_corpus
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


if __name__ == '__main__':

    load_dotenv()

    problem_type = 'binary'
    # problem_type = 'multi_class'

    n_execution = 2

    corpus_file_path = '../data/corpus/general/corpus_grammar_spell_errors.json'

    outputs_dir = f'../data/experiments/classification/outputs/{problem_type}/{n_execution}'
    results_dir = f'../data/experiments/classification/results/{problem_type}/llms/{n_execution}'

    if problem_type == 'binary':
        prompt_file_path = '../../data/prompts/prompt_classificacao_binaria.txt'
    else:
        prompt_file_path = '../../data/prompts/prompt_classificacao_multiclasse.txt'

    SABIA_API_KEY = os.getenv('SABIA_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TOGETHER_AI_API_KEY = os.getenv('TOGETHER_AI_API_KEY')
    DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')

    n_samples = -1

    max_seq_length = 1024

    max_new_tokens = 10

    print('\nReading Corpus')

    with open(file=corpus_file_path, mode='r', encoding='utf-8') as file:
        corpus = json.load(file)

    if n_samples > 0:
        corpus = corpus[:n_samples]

    print(f'\n\tTotal of Sentences: {len(corpus)}')

    with open(file=prompt_file_path, mode='r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()

    if problem_type == 'multi_class':
        list_texts, list_labels = build_multi_class_corpus(corpus)
    else:
        list_texts, list_labels = build_binary_corpus(corpus)

    print(f'\n\tTotal Examples: {len(list_texts)} -- {len(list_labels)}')

    # llm_model_dict = {
    #     'model_name': 'sabiazinho_3',
    #     'type': 'maritaca',
    #     'url_base': 'https://chat.maritaca.ai/api',
    #     'model_checkpoint': 'sabiazinho-3'
    # }

    # llm_model_dict = {
    #     'model_name': 'gemma_3_27b',
    #     'type': 'deepinfra',
    #     'url_base': 'https://api.deepinfra.com/v1/openai',
    #     'model_checkpoint': 'google/gemma-3-27b-it'
    # }

    # llm_model_dict = {
    #     'model_name': 'qwen_3_235b_a22b',
    #     'type': 'deepinfra',
    #     'url_base': 'https://api.deepinfra.com/v1/openai',
    #     'model_checkpoint': 'Qwen/Qwen3-235B-A22B-Instruct-2507'
    # }

    llm_model_dict = {
        'model_name': 'llama_4_maverick',
        'type': 'deepinfra',
        'url_base': 'https://api.deepinfra.com/v1/openai',
        'model_checkpoint': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
    }

    # llm_model_dict = {
    #     'model_name': 'gemini_25_flash',
    #     'type': 'deepinfra',
    #     'url_base': 'https://api.deepinfra.com/v1/openai',
    #     'model_checkpoint': 'google/gemini-2.5-flash'
    # }

    print('\nLoading Model')

    if llm_model_dict['type'] == 'openai':
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        delay = 1
    elif llm_model_dict['type'] == 'together_ai':
        client = OpenAI(
            base_url=llm_model_dict['url_base'],
            api_key=TOGETHER_AI_API_KEY
        )
        delay = 2
    elif llm_model_dict['type'] == 'deepinfra':
        client = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url=llm_model_dict['url_base']
        )
        delay = 2
    else:
        client = OpenAI(
            base_url=llm_model_dict['url_base'],
            api_key=SABIA_API_KEY
        )
        delay = 1

    print(f'\n{50 * "="} RUNNING EXPERIMENT {problem_type} -- {llm_model_dict["model_name"]} {50 * "="}\n')

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    outputs_file_path = os.path.join(outputs_dir, f'{llm_model_dict["model_name"]}.json')

    generated_outputs = []
    set_input_texts_processed = []

    y_true = []
    y_pred = []

    if os.path.exists(outputs_file_path):
        with open(file=outputs_file_path, mode='r', encoding='utf-8') as file:
            try:
                generated_outputs = json.load(file)
                set_input_texts_processed = set([e['texto'] for e in generated_outputs])
                y_true = [e['label_real'] for e in generated_outputs]
                y_pred = [e['label_predita'] for e in generated_outputs]
            except json.JSONDecodeError:
                pass

    with tqdm(total=len(list_texts), file=sys.stdout, colour='blue', desc='\tRunning') as pbar:

        for input_text, real_label in zip(list_texts, list_labels):

            if input_text in set_input_texts_processed:
                pbar.update(1)
                continue

            prompt = prompt_template.replace('{texto_entrada}', input_text)

            input_messages = [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]

            response = client.chat.completions.create(
                model=llm_model_dict['model_checkpoint'],
                messages=input_messages,
                temperature=0.1,
                top_p=0.1
            )

            estimated_label = response.choices[0].message.content.strip()

            if 'SAÍDA:' in response:
                estimated_label = estimated_label.split('SAÍDA:')[-1].strip()

            estimated_label = estimated_label.replace('.', '').strip()

            if 'CORRETA' in estimated_label:
                estimated_label = 'CORRETA'
            elif 'ERRO' in estimated_label:
                estimated_label = 'ERRO'
            elif 'ORTOGRÁFICO' in estimated_label:
                estimated_label = 'ORTOGRÁFICO'
            elif 'GRAMATICAL' in estimated_label:
                estimated_label = 'GRAMATICAL'

            generated_outputs.append(
                {
                    'texto': input_text,
                    'label_real': real_label,
                    'label_predita': estimated_label,
                }
            )

            with open(file=outputs_file_path, mode='w', encoding='utf-8') as output_file:
                json.dump(obj=generated_outputs, fp=output_file, indent=4, ensure_ascii=False)

            y_true.append(real_label)
            y_pred.append(estimated_label)

            pbar.update(1)

            time.sleep(delay)

    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    classification_report_file_name = f'{llm_model_dict["model_name"]}_report.json'.lower()

    classification_report_file_path = os.path.join(results_dir, classification_report_file_name)

    with open(file=classification_report_file_path, mode='w') as file_:
        json.dump(report_dict, file_, indent=4)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    confusion_matrix_name = f'{llm_model_dict["model_name"]}_confusion_matrix.pdf'.lower()

    img_path = os.path.join(results_dir, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)
