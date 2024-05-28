import os
import json
import openai
import argparse
from tqdm import tqdm
from rich import print as rprint
from datasets import load_dataset

from utils import (
    preprocess_generation,
    call_openai_api,
    call_opensource_model,
)
from methods.self_refine.self_refine import self_refine
from methods.rag.rag import RAGCutoff
from methods.narrative_experts.narrative_experts import (
    narrative_experts,
    NarrativeExpertsRAGCutoff,
)

openai.api_key = os.getenv('OPENAI_API_KEY')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='gpt-4-1106-preview', choices=['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'llama-2-13b-chat', 'mistral-instruct-7b'])
    parser.add_argument("--method_name", default='zero-shot', choices=['zero-shot','zero-shot-cot', 'few-shot', 'self-refine', 'rag-cutoff', 'narrative-experts', 'narrative-experts-rag-cutoff'])
    parser.add_argument("--data_split", default='validation', choices=['validation', 'test'])
    parser.add_argument("--output_dir", default='outputs/', type=str, help="output directoy")
    parser.add_argument("--output_fname", default='generated.json', type=str, help="output file name")
    parser.add_argument("--rag_cache_dir", default='methods/rag/text-embedding-ada-002/cutoff', type=str, help="RAG cache directory")
    args = parser.parse_args()

    # Load TimeChara
    dataset = load_dataset('ahnpersie/timechara')[args.data_split]
    outputs = []

    # Load Model
    if args.model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"
    elif args.model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Warning: The user must have permission to access gated repositories.
        if args.model_name == 'llama-2-13b-chat':
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype="auto").to(device=device)
        elif args.model_name == 'mistral-instruct-7b':
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device=device)
        else:
            raise NotImplementedError
        model.eval()
    else:
        raise NotImplementedError

    # Load requirements
    if args.method_name == 'few-shot':
        few_shot_dict = dict()
        for series_name in ['harry_potter', 'the_lord_of_the_rings', 'twilight', 'hunger_games']:
            with open(f'methods/few_shot/{series_name}.json', 'r') as fp:
                current_dict = json.load(fp)
            few_shot_dict.update(current_dict)
    elif args.method_name == 'rag-cutoff':
        assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"
        rag = RAGCutoff(openai.api_key, args.rag_cache_dir)
    elif args.method_name == 'narrative-experts-rag-cutoff':
        assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"
        rag = NarrativeExpertsRAGCutoff(openai.api_key, args.rag_cache_dir)

    data_cnt = 0
    for example in tqdm(dataset, ncols=120):
        series = example['series']
        data_type = example['data_type']
        character = example['character']
        character_period = example['character_period']
        question = example['question']
        question_period = example['question_period']
        participants = example['participants']

        system_prompt, book_no, day, day_character = preprocess_generation(series, character, character_period)

        if args.model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
            if args.method_name == 'zero-shot':
                completion = call_openai_api(args.model_name, system_prompt, question,
                                             max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
                response = completion.choices[0].message.content
                hint = '-'
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of response tokens = {completion.usage.completion_tokens}")
                rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\n[green]({data_type})Question: {question}[/green]\n\n[bold green]Response: {response}[/bold green]")
            elif args.method_name == 'zero-shot-cot':
                question_zero_shot_cot = f"{question}\nLet's think step by step."
                completion = call_openai_api(args.model_name, system_prompt, question_zero_shot_cot,
                                             max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
                response = completion.choices[0].message.content
                hint = '-'
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of response tokens = {completion.usage.completion_tokens}")
                rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\n[green]({data_type})Question: {question_zero_shot_cot}[/green]\n\n[bold green]Response: {response}[/bold green]")
            elif args.method_name == 'few-shot':
                example = few_shot_dict[f'{character}-{book_no}-{day}']
                question_few_shot = f"{example}\n***\nQuestion: {question}\nResponse: "
                completion = call_openai_api(args.model_name, system_prompt, question_few_shot,
                                             max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
                response = completion.choices[0].message.content
                hint = '-'
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of response tokens = {completion.usage.completion_tokens}")
                rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\nExamples: {example}\n\n[green]({data_type}){question_few_shot}[/green]\n\n[bold green]Response: {response}[/bold green]")
            elif args.method_name == 'self-refine':
                response, hint = self_refine(args.model_name, system_prompt, question, day_character,
                                             data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
            elif args.method_name == 'rag-cutoff':
                response, hint = rag.generate(args.model_name, system_prompt, question, character, series, book_no, day,
                                                data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
            elif args.method_name == 'narrative-experts':
                response, hint = narrative_experts(args.model_name, system_prompt, question, question_period, character, book_no, day, participants, series,
                                                   data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
            elif args.method_name == 'narrative-experts-rag-cutoff':
                response, hint = rag.generate(args.model_name, system_prompt, question, question_period, character, book_no, day, participants, series,
                                              use_rag_for_temporal_expert=True, use_rag_for_spatial_expert=True,
                                              data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
            else:
                raise NotImplementedError
        elif args.model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
            if args.method_name == 'zero-shot':
                if args.model_name in ['llama-2-13b-chat']:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]
                elif args.model_name in ['mistral-instruct-7b']:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n***\nQuestion: {question}\nResponse:"},
                    ]
                else:
                    raise NotImplementedError
                response = call_opensource_model(messages, model, tokenizer, max_tokens=2048, temperature=0.2, top_p=1.0, device=device)
                hint = '-'
                rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\n[green]({data_type})Question: {question}[/green]\n\n[bold green]Response: {response}[/bold green]")
            elif args.method_name == 'zero-shot-cot':
                question_zero_shot_cot = f"{question}\nLet's think step by step."
                if args.model_name in ['llama-2-13b-chat']:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{question_zero_shot_cot}"},
                    ]
                elif args.model_name in ['mistral-instruct-7b']:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n***\nQuestion: {question_zero_shot_cot}\nResponse:"},
                    ]
                else:
                    raise NotImplementedError
                response = call_opensource_model(messages, model, tokenizer, max_tokens=2048, temperature=0.2, top_p=1.0, device=device)
                hint = '-'
                rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\n[green]({data_type})Question: {question_zero_shot_cot}[/green]\n\n[bold green]Response: {response}[/bold green]")
            elif args.method_name == 'few-shot':
                example = few_shot_dict[f'{character}-{book_no}-{day}']
                question_few_shot = f"{example}\n***\nQuestion: {question}\nResponse: "
                if args.model_name in ['llama-2-13b-chat']:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question_few_shot},
                    ]
                elif args.model_name in ['mistral-instruct-7b']:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n***\n{example}\n***\nQuestion: {question}\nResponse:"},
                    ]
                else:
                    raise NotImplementedError                
                response = call_opensource_model(messages, model, tokenizer, max_tokens=2048, temperature=0.2, top_p=1.0, device=device)
                hint = '-'
                rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\nExamples: {example}\n\n[green]({data_type}){question_few_shot}[/green]\n\n[bold green]Response: {response}[/bold green]")
            elif args.method_name == 'self-refine':
                response, hint = self_refine(args.model_name, system_prompt, question, day_character, tokenizer=tokenizer, model=model,
                                             data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0, device=device)
            elif args.method_name == 'rag-cutoff':
                response, hint = rag.generate(args.model_name, system_prompt, question, character, series, book_no, day, tokenizer=tokenizer, model=model,
                                              data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0, device=device)
            elif args.method_name == 'narrative-experts':
                response, hint = narrative_experts(args.model_name, system_prompt, question, question_period, character, book_no, day, participants, series, tokenizer=tokenizer, model=model,
                                                   data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0, device=device)
            elif args.method_name == 'narrative-experts-rag-cutoff':
                response, hint = rag.generate(args.model_name, system_prompt, question, question_period, character, book_no, day,
                                              participants, series, use_rag_for_temporal_expert=True, use_rag_for_spatial_expert=True, tokenizer=tokenizer, model=model,
                                              data_type=data_type, max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0, device=device)
            else:
                raise ValueError
        else:
            raise NotImplementedError
        
        outputs.append({
            'data_idx': data_cnt,
            'response': response,
            'thought': hint,
        })
        data_cnt += 1

        # save temporary outputs
        if data_cnt % 10 == 0:
            os.makedirs(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}'), exist_ok=True)
            with open(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'{args.output_fname}.temp'), 'w') as fp:
                json.dump(outputs, fp, indent=4)
            print(f"\n\nSaved outputs to {os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', f'{args.output_fname}.temp')}!!\n\n")

    # save outputs
    os.makedirs(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}'), exist_ok=True)
    with open(os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', args.output_fname), 'w') as fp:
        json.dump(outputs, fp, indent=4)
    print(f"\n\nSaved outputs to {os.path.join(args.output_dir, args.data_split, f'{args.model_name}_{args.method_name}', args.output_fname)}!!\n\n")

    print('Good Job Computer!')

if __name__ == '__main__':
    main()