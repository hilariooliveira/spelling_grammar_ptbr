import os
import json
import torch
import sys
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from exp_utils import build_binary_corpus, build_multi_class_corpus
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


if __name__ == '__main__':

    load_dotenv()

    # problem_type = 'binary'
    problem_type = 'multi_class'

    n_execution = 3

    corpus_file_path = '../data/corpus/general/corpus_grammar_spell_errors.json'

    outputs_dir = f'../data/experiments/classification/outputs/{problem_type}/{n_execution}'
    results_dir = f'../data/experiments/classification/results/{problem_type}/llms/{n_execution}'

    if problem_type == 'binary':
        prompt_file_path = '../data/prompts/prompt_classificacao_binaria.txt'
    else:
        prompt_file_path = '../data/prompts/prompt_classificacao_multiclasse.txt'

    SABIA_API_KEY = os.getenv('SABIA_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TOGETHER_AI_API_KEY = os.getenv('TOGETHER_AI_API_KEY')

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
    #     'model_name': 'llama_31_8b',
    #     'type': 'unsloth',
    #     'model_checkpoint': 'unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit'
    # }

    llm_model_dict = {
        'model_name': 'gemma_2_9b',
        'type': 'unsloth',
        'model_checkpoint': 'unsloth/gemma-2-9b-it-bnb-4bit'
    }

    # llm_model_dict = {
    #     'model_name': 'qwen_25_7b',
    #     'type': 'unsloth',
    #     'model_checkpoint': 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'
    # }

    # llm_model_dict = {
    #     'model_name': 'qwen_25_14b',
    #     'type': 'unsloth',
    #     'model_checkpoint': 'unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit'
    # }

    print('\nLoading Model')

    if llm_model_dict['type'] == 'transformers':

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            llm_model_dict['model_checkpoint'],
            quantization_config=quantization_config
        )

    elif llm_model_dict['type'] == 'unsloth':

        model = AutoModelForCausalLM.from_pretrained(
            llm_model_dict['model_checkpoint']
        )

    else:
        print('\nERROR')
        exit(-1)

    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_dict['model_checkpoint']
    )

    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        return_full_text=False,
    )

    llm_pipeline = HuggingFacePipeline(
        pipeline=pipe,
        model_id=llm_model_dict['model_checkpoint']
    )

    llm_chat_model = ChatHuggingFace(llm=llm_pipeline)

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

            if 'gemma' in llm_model_dict['model_name']:
                messages = [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': prompt
                            }
                        ]
                    },
                ]
            else:
                messages = [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]

            estimated_label = llm_chat_model.invoke(messages)

            estimated_label = estimated_label.content.strip()

            if 'SAÍDA:' in estimated_label:
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

    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    classification_report_file_name = f'{llm_model_dict["model_name"]}_report.json'.lower()

    classification_report_file_path = os.path.join(results_dir, classification_report_file_name)

    with open(file=classification_report_file_path, mode='w') as file_:
        json.dump(report_dict, file_, indent=4)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    confusion_matrix_name = f'{llm_model_dict["model_name"]}_confusion_matrix.pdf'.lower()

    img_path = os.path.join(results_dir, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)
