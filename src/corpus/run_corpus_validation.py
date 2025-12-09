import os
import json

from dotenv import load_dotenv
from openai import OpenAI


if __name__ == '__main__':

    load_dotenv()

    corpus_file_path = '../../data/corpus/general/general_corpus_grammar_spell_errors.json'
    validated_corpus_file_path = '../../data/corpus/general/validated_general_corpus_grammar_spell_errors.json'

    prompt_file_path = '../../data/prompts/prompt_validacao_corpus.txt'

    model_checkpoint = 'sabia-3.1'

    SABIA_API_KEY = os.getenv('SABIA_API_KEY')

    print('\nReading Corpus')

    with open(file=corpus_file_path, mode='r', encoding='utf-8') as file:
        corpus = json.load(file)

    with open(file=prompt_file_path, mode='r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()

    print(f'\n\tTotal of Sentences: {len(corpus)}')

    openai = OpenAI(
        api_key=SABIA_API_KEY,
        base_url='https://chat.maritaca.ai/api',
    )

    print('\nRunning Validation\n')

    validated_corpus = []

    set_sentences_processed = set()

    if os.path.exists(validated_corpus_file_path):
        with open(file=validated_corpus_file_path, mode='r', encoding='utf-8') as file:
            try:
                validated_corpus = json.load(file)
                set_sentences_processed = set([e['frase_com_erros'] for e in validated_corpus])
            except json.JSONDecodeError:
                pass

    for cont, entry in enumerate(corpus, start=1):

        print(f'  Sentence {cont} / {len(corpus)}: {entry["frase_com_erros"]}')
        # print(f'    Erros: {str(entry["erros"])}')

        if entry["frase_com_erros"] in set_sentences_processed:
            continue

        prompt = prompt_template.replace('{frase}', entry['frase_com_erros'])
        prompt = prompt.replace('{erros}', str(entry['erros']))

        input_messages = [
            {
                'role': 'user',
                'content': prompt
            }
        ]

        response = openai.chat.completions.create(
            model=model_checkpoint,
            messages=input_messages,
            temperature=0.1,
            top_p=0.1
        )

        response = response.choices[0].message.content

        response = response.replace('\n', ' ')
        label = response.split('LABEL:')[-1]
        label = label.split('JUSTIFICATIVA:')[0]

        entry['VALIDATION_LABEL'] = label.strip()

        entry['VALIDATION_OUTPUT'] = response

        validated_corpus.append(entry)

        with open(file=validated_corpus_file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(obj=validated_corpus, fp=output_file, indent=4, ensure_ascii=False)
