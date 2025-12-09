import os
import pandas as pd
import sys
import json

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


if __name__ == '__main__':

    load_dotenv()

    corpus_path = '../../../data/corpus/general/general_dataset.csv'

    corpus_file_path = '../../../data/corpus/general/'

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    prompt_file_path = '../../../data/prompts/prompt_corpus_generation.txt'

    os.makedirs(corpus_file_path, exist_ok=True)

    corpus_file_path = os.path.join(corpus_file_path, 'general_corpus_grammar_spell_errors.json')

    model_checkpoint = 'gpt-4o-mini'

    dataset = pd.read_csv(filepath_or_buffer=corpus_path)

    print(f'\nTotal Sentences: {len(dataset)}')

    with open(file=prompt_file_path, mode='r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    print(f'\n{55 * "="} Generating Corpus {55 * "="}\n')

    corpus = []

    set_sentences = set()

    if os.path.exists(corpus_file_path):
        with open(file=corpus_file_path, mode='r', encoding='utf-8') as file:
            try:
                corpus = json.load(file)
                set_sentences = set([e['frase_original'] for e in corpus])
            except json.JSONDecodeError:
                pass

    with tqdm(total=len(dataset), file=sys.stdout, colour='blue', desc='\tGenerating corpus') as pbar:

        for index, row in dataset.iterrows():

            sentence = row['Sentença Corrigida']

            if sentence in set_sentences:
                pbar.update(1)
                continue

            prompt = prompt_template.replace('{lista_sentencas}', str([sentence]))

            input_messages = [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]

            response = client.chat.completions.create(
                model=model_checkpoint,
                messages=input_messages,
                temperature=0.1,
                top_p=0.1
            )

            response = response.choices[0].message.content

            errors = response.replace('```python', '')
            errors = errors.replace('```', '')

            errors = eval(errors)

            corpus.extend(errors)

            with open(file=corpus_file_path, mode='w', encoding='utf-8') as output_file:
                json.dump(obj=corpus, fp=output_file, indent=4, ensure_ascii=False)

            pbar.update(1)
