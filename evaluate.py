import os
import copy
import json
import openai
import argparse
from tqdm import tqdm
from rich import print as rprint
from datasets import load_dataset

from utils import (
    character_dict,
    preprocess_evaluation,
    call_openai_api,
    extract_score,
)

openai.api_key = os.getenv('OPENAI_API_KEY')

def print_spatiotemporal_scores(dataset, outputs):
    assert len(dataset) == len(outputs), f"Error: # dataset != # generated outputs"
    spatiotemporal_score = {'future': [], 'past-absence': [], 'past-presence': [], 'past-only': []}
    for data_idx, example in enumerate(dataset):
        output = outputs[data_idx]
        data_type = example['data_type']
        assert output['temporal_score'] != ''
        if data_type in ['future', 'past-only']:
            spatiotemporal_score[data_type].append(output['temporal_score'])
        elif data_type in ['past-absence', 'past-presence']:
            assert output['spatial_score'] != ''
            spatiotemporal_score[data_type].append(output['temporal_score'] * output['spatial_score'])
        else:
            raise ValueError

    combined_scores = []
    for value in spatiotemporal_score.values():
        combined_scores.extend(value)

    print(f'\n*** Spatiotemporal Scores ***\n')
    for k,v in spatiotemporal_score.items():
        if v == []:
            continue
        else:
            print(f'{k} (max 1.0): {sum(v)}/{len(v)} (={round(sum(v)/len(v),5)})')
    print(f'\nTotal (max 1.0): {sum(combined_scores)}/{len(combined_scores)} (={round(sum(combined_scores)/len(combined_scores),5)})')

def print_personality_scores(dataset, outputs):
    assert len(dataset) == len(outputs), f"Error: # dataset != # generated outputs"
    combined_scores = []
    for data_idx, example in enumerate(dataset):
        output = outputs[data_idx]
        assert output['personality_score'] != ''
        combined_scores.append(output['personality_score'])

    print(f'\n*** Personality Scores ***\n')
    print(f'\nTotal (max 7.0): {sum(combined_scores)}/{len(combined_scores)} (={round(sum(combined_scores)/len(combined_scores),5)})')

def evaluate_spatiotemporal_consistency(model_name, spatiotemporal_prompt_template, agent_name, question, answer, agent_fact):
    prompt = spatiotemporal_prompt_template.format(agent_name=agent_name,
                                                   question_0=question,
                                                   answer_0=answer,
                                                   agent_fact_0=agent_fact)
    completion = call_openai_api(model_name, 'You are a helpful and accurate assistant.', prompt)
    print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
    print(f"# of completion tokens = {completion.usage.completion_tokens}")
    content = completion.choices[0].message.content
    rprint(f"[blue]Question: {question}[/blue]\n[green]{agent_name}: {answer}[/green]\n\n[bold green]Evaluation:{content}[/bold green]")
    return content

def evaluate_personality_consistency(model_name, personality_prompt_template, agent_name, question, answer, agent_personality):
    prompt = personality_prompt_template.format(agent_name=agent_name,
                                                question_0=question,
                                                response_0=answer,
                                                agent_personality=agent_personality)
    completion = call_openai_api(model_name, 'You are a helpful and accurate assistant.', prompt)
    print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
    print(f"# of completion tokens = {completion.usage.completion_tokens}")
    content = completion.choices[0].message.content
    rprint(f"[blue]Question: {question}[/blue]\n[green]{agent_name}: {answer}[/green]\n\n[bold green]Evaluation:{content}[/bold green]")
    return content

def main():
    assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='gpt-4-1106-preview', choices=['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'llama-2-13b-chat', 'mistral-instruct-7b',])
    parser.add_argument("--eval_model_name", default='gpt-4-1106-preview', choices=['gpt-4-1106-preview', 'gpt-3.5-turbo-1106'])
    parser.add_argument("--method_name", default='zero-shot', choices=['zero-shot','zero-shot-cot', 'few-shot', 'self-refine', 'rag-cutoff', 'narrative-experts', 'narrative-experts-rag-cutoff'])
    parser.add_argument("--data_split", default='validation', choices=['validation', 'test'])
    parser.add_argument("--eval_mode", default='spatiotemporal', choices=['spatiotemporal', 'personality', 'all'])
    parser.add_argument("--output_dir", default='outputs/', type=str, help="output directoy")
    parser.add_argument("--input_fname", default='generated.json', type=str, help="input file name ")
    parser.add_argument("--output_fname", default='evaluated.json', type=str, help="output file name")
    args = parser.parse_args()

    # Load TimeChara
    dataset = load_dataset('ahnpersie/timechara')[args.data_split]
    with open(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', args.input_fname), 'r') as fp:
        responses = json.load(fp)
    assert len(dataset) == len(responses), f"Error: # dataset != # generated responses"
    outputs = []

    personality = dict.fromkeys(character_dict, '')
    for k,v in character_dict.items():
        with open(f'data/personality/{v}_personality.txt', 'r', encoding='utf-8') as fp:
            personality[k] = fp.read()

    with open('data/spatiotemporal_consistency_evaluation_prompt.txt', 'r') as fp:
        spatiotemporal_prompt_template = fp.read()
    with open('data/personality_consistency_evaluation_prompt.txt', 'r') as fp:
        personality_prompt_template = fp.read()

    data_cnt = 0
    assert args.eval_mode in ['spatiotemporal', 'personality', 'all']
    for data_idx, example in tqdm(enumerate(dataset), total=len(dataset), ncols=120):
        assert data_idx == responses[data_idx]['data_idx'], f"Error: data_idx mismatch!"
        series = example['series']
        question = example['question']
        response = responses[data_idx]['response']
        hint = responses[data_idx]['thought']
        character = example['character']
        character_period = example['character_period']

        output = {
            'data_idx': data_cnt,
            'response': response,
            'thought': hint,
            'temporal_eval': '',
            'temporal_score': '',
            'spatial_eval': '',
            'spatial_score': '',
            'personality_eval': '',
            'personality_score': '',
        }

        day_character = preprocess_evaluation(series, character, character_period)

        # spatiotemporal consistency
        if args.eval_mode in ['all', 'spatiotemporal']:
            data_type = example['data_type']
            temporal_label = example['temporal_label']
            spatial_label = example['spatial_label']
            assert data_type in ['future', 'past-absence', 'past-presence', 'past-only']

            if 'future' in data_type:
                assert temporal_label.split(':')[0].lower() == 'future'
                assert spatial_label == '-'
                content_temporal = evaluate_spatiotemporal_consistency(
                    args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, temporal_label)
                temporal_consistency_score, _ = extract_score(content_temporal)
                output['temporal_eval'] = copy.deepcopy(content_temporal)
                output['temporal_score'] = copy.deepcopy(temporal_consistency_score)
            elif 'past' in data_type:
                assert temporal_label.split(':')[0].lower() == 'past'
                if data_type in ['past-absence', 'past-presence']:
                    assert spatial_label != '-'
                    content_spatial = evaluate_spatiotemporal_consistency(
                        args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, spatial_label)
                    spatial_consistency_score, _ = extract_score(content_spatial)
                    # check spatial consistency
                    if spatial_consistency_score == '0':
                        content_temporal = 'Spatiotemporally inconsistent'
                        temporal_consistency_score = '0'
                    else:
                        content_temporal = evaluate_spatiotemporal_consistency(
                            args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, temporal_label)
                        temporal_consistency_score, _ = extract_score(content_temporal)
                    output['spatial_eval'] = copy.deepcopy(content_spatial)
                    output['spatial_score'] = copy.deepcopy(spatial_consistency_score)
                    output['temporal_eval'] = copy.deepcopy(content_temporal)
                    output['temporal_score'] = copy.deepcopy(temporal_consistency_score)
                elif data_type == 'past-only':
                    assert spatial_label == '-'
                    content_temporal = evaluate_spatiotemporal_consistency(
                        args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, temporal_label)
                    temporal_consistency_score, _ = extract_score(content_temporal)
                    output['temporal_eval'] = copy.deepcopy(content_temporal)
                    output['temporal_score'] = copy.deepcopy(temporal_consistency_score)
                else:
                    raise ValueError
            else:
                raise ValueError

        # personality consistency
        if args.eval_mode in ['all', 'personality']:
            content_personality = evaluate_personality_consistency(
                args.eval_model_name, personality_prompt_template, day_character, question, response, personality[character])
            personality_consistency_score, _ = extract_score(content_personality)
            output['personality_eval'] = copy.deepcopy(content_personality)
            output['personality_score'] = copy.deepcopy(personality_consistency_score)

        outputs.append(output)
        data_cnt += 1

        # save temporary outputs
        if data_cnt % 10 == 0:
            os.makedirs(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}'), exist_ok=True)
            with open(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}', f'{args.output_fname}.temp'), 'w') as fp:
                json.dump(outputs, fp, indent=4)
            print(f"\n\nSaved outputs to {os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}', f'{args.output_fname}.temp')}!!\n\n")

    # save outputs
    os.makedirs(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}'), exist_ok=True)
    with open(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}', args.output_fname), 'w') as fp:
        json.dump(outputs, fp, indent=4)
    print(f"\n\nSaved outputs to {os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}', args.output_fname)}!!\n\n")

    # print scores
    if args.eval_mode in ['spatiotemporal', 'all']:
        print_spatiotemporal_scores(dataset, outputs)
    if args.eval_mode in ['personality', 'all']:
        print_personality_scores(dataset, outputs)

    print('Good Job Computer!')

if __name__ == '__main__':
    main()
