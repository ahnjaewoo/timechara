from rich import print as rprint

from utils import (
    series_name_dict,
    character_period,
    call_openai_api,
    call_opensource_model,
    extract_and_format_number,
    compare_book_chapters,
)
from methods.rag.rag import (
    RAGCutoff,
)

def narrative_experts(model_name,
                      system_prompt,
                      question,
                      question_period,
                      character,
                      book_no,
                      day,
                      participants,
                      series_name,
                      tokenizer=None,
                      model=None,
                      data_type=None,
                      max_tokens=1024,
                      temperature=0.0,
                      top_p=0.95,
                      n=1,
                      seed=0,
                      device='cpu'):
    hint = []

    if series_name == 'harry_potter':
        try:
            character_book_chapter_num_abs = character_period[series_name_dict[series_name]][day.lower()][int(book_no) - 1]
        except:
            character_book_chapter_num_abs = f'{book_no}-0'
        book_chapter_name = f'book number and chapter number'
        book_chapter_format = f'book M - chapter N (write Arabic number instead of Roman number for the volume number)'
        book_chapter_cnt = 2
    elif series_name == 'the_lord_of_the_rings':
        if day == 'at the end of the scene':
            character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][f'end of Volume{book_no}']
        else:
            character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][day]
        book_chapter_name = f'volume number, book number, and chapter number'
        book_chapter_format = f'volume L - book M - chapter N (write Arabic number instead of Roman number for the volume number)'
        book_chapter_cnt = 3
    elif series_name in ['twilight', 'hunger_games']:
        if day == 'at the end of the scene':
            character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][f'end of book{book_no}']
        else:
            character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][day]
        book_chapter_name = f'book number and chapter number'
        book_chapter_format = f'book M - chapter N (write Arabic number instead of book title for the book number)'
        book_chapter_cnt = 2
    else:
        raise ValueError
    series_full_name = series_name_dict[series_name]

    user_prompt = f"""You will be given a question from {series_full_name} series at a specific time. Your task is to identify the exact {book_chapter_name} of the scene of the question. Below is the data:
***
[Question]
{question}
***
[Identification Criterion]
What is the exact {book_chapter_name} of the scene of the question?

[Identification Steps]
1. Read through the [Question], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question, in '{book_chapter_format}' format.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""

    if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        completion = call_openai_api(model_name, "You are a helpful and accurate assistant.", f"{user_prompt}",
                                     max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
        content = completion.choices[0].message.content
        print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
        print(f"# of completion tokens = {completion.usage.completion_tokens}")
    elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
        assert tokenizer is not None and model is not None and data_type is not None
        if model_name in ['llama-2-13b-chat']:
            messages = [
                {"role": "system", "content": "You are a helpful and accurate assistant."},
                {"role": "user", "content": user_prompt},
            ]
        elif model_name in ['mistral-instruct-7b']:
            messages = [
                {"role": "user", "content": user_prompt},
            ]
        else:
            raise NotImplementedError
        content = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
    else:
        raise NotImplementedError
    rprint(f"[purple]Prompt: {user_prompt}[/purple]\n***\n[bold purple]Response: {content}\n***[/bold purple]")
    rprint(f"[bold purple]Gold question period: {question_period}[/bold purple]")

    temporal_label_str = extract_and_format_number(content.strip().split('\n')[-1].lower(), book_chapter_cnt)

    is_future = False
    if temporal_label_str == '':
        pass
    else:
        temporal_label = compare_book_chapters(temporal_label_str, character_book_chapter_num_abs)
        if temporal_label == 'after':
            hint.append(f"Note that the period of the question is in the future relative to {character}'s time point. Therefore, you should not answer the question or mention any facts that occurred after {character}'s time point.")
            is_future = True
        elif temporal_label == 'before':
            # hint.append(f"Note that the period of the question is in the past relative to {character}'s time point. Therefore, you can respond based on your past experience.")
            pass
        else:
            raise ValueError

    if is_future:
        pass
    else:
        user_prompt = f"""You will be given a question and a character from {series_full_name} series. Your task is to classify whether the character is a participant (i.e., present or absent) in the scene of the question. Below is the data:
***
[Question]
{question}
[Character]
{character}
***
[Classification Criterion]
Is the character a participant in the scene of the question?

[Classification Steps]
1. Read through the [Question], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question.
3. Write a list of every character involved in the scene described in the question, including those not explicitly mentioned in the question but who were present in the scene.
4. Compare the list of participants to the character. Check if the list of participants contains the character.
5. If the list contains the character, classify it as 'present'. Otherwise, classify it as 'absent'.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""

    if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        completion = call_openai_api(model_name, "You are a helpful and accurate assistant.", f"{user_prompt}",
                                        max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
        content = completion.choices[0].message.content
        print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
        print(f"# of completion tokens = {completion.usage.completion_tokens}")
    elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
        assert tokenizer is not None and model is not None and data_type is not None
        if model_name in ['llama-2-13b-chat']:
            messages = [
                {"role": "system", "content": "You are a helpful and accurate assistant."},
                {"role": "user", "content": user_prompt},
            ]
        elif model_name in ['mistral-instruct-7b']:
            messages = [
                {"role": "user", "content": user_prompt},
            ]
        else:
            raise NotImplementedError
        content = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
    else:
        raise NotImplementedError
    rprint(f"[purple]Prompt: {user_prompt}[/purple]\n***\n[bold purple]Response: {content}[/bold purple]")
    rprint(f"[bold purple]Gold participants: {participants}[/bold purple]")

    spatial_label_str = content.strip().split('\n')[-1].lower()
    if 'present' in spatial_label_str and 'absent' in spatial_label_str:
        pass
    elif 'present' in spatial_label_str:
        # hint.append(f"Note that {character} had participated in the scene described in the question. Therefore, you should not imply that {character} was absent in the scene.")
        pass
    elif 'absent' in spatial_label_str:
        hint.append(f"Note that {character} had not participated in the scene described in the question. Therefore, you should not imply that {character} was present in the scene.")
    else:
        pass

    # Final response
    if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview'] + ['llama-2-13b-chat']:
        final_query = question if len(hint) == 0 else f"{question}\n(HINT: {' '.join(hint)})"
    elif model_name in ['gpt-3.5-turbo-1106']:
        final_query = question if len(hint) == 0 else f"(HINT: {' '.join(hint)})\n{question}"
    elif model_name == 'mistral-instruct-7b':
        final_query = question if len(hint) == 0 else f"Question: {question}\n(HINT: {' '.join(hint)})\nResponse:"
    else:
        raise NotImplementedError

    if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        completion = call_openai_api(model_name, system_prompt, final_query,
                                     max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
        response = completion.choices[0].message.content
        print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
        print(f"# of completion tokens = {completion.usage.completion_tokens}")
    elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
        assert tokenizer is not None and model is not None and data_type is not None
        if model_name in ['llama-2-13b-chat']:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_query},
            ]
        elif model_name in ['mistral-instruct-7b']:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n***\n{final_query}"},
            ]
        else:
            raise NotImplementedError
        response = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
    else:
        raise NotImplementedError
    rprint(f"[blue]Prompt: {system_prompt}[/blue]\n***\n[green]({data_type}) {final_query}[/green]\n\n[bold green]Final Response: {response}\n***[/bold green]")

    return response, '\n'.join(hint)

class NarrativeExpertsRAGCutoff(RAGCutoff):
    def __init__(self,
                 openai_api_key,
                 rag_cache_dir):
        super().__init__(openai_api_key, rag_cache_dir)

    def generate(self,
                 model_name,
                 system_prompt,
                 question,
                 question_period,
                 character,
                 book_no,
                 day,
                 participants,
                 series_name,
                 use_rag_for_temporal_expert=False,
                 use_rag_for_spatial_expert=False,
                 tokenizer=None,
                 model=None,
                 data_type=None,
                 max_tokens=1024,
                 temperature=0.0,
                 top_p=0.95,
                 n=1,
                 seed=0,
                 device='cpu'):
        hint = []

        if series_name == 'harry_potter':
            try:
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][day.lower()][int(book_no) - 1]
            except:
                character_book_chapter_num_abs = f'{book_no}-0'
            book_chapter_name = f'book number and chapter number'
            book_chapter_format = f'book M - chapter N (write Arabic number instead of book title for the book number)'
            book_chapter_cnt = 2
        elif series_name == 'the_lord_of_the_rings':
            if day == 'at the end of the scene':
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][f'end of Volume{book_no}']
            else:
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][day]
            book_chapter_name = f'volume number, book number, and chapter number'
            book_chapter_format = f'volume L - book M - chapter N (write Arabic number instead of Roman number for the volume number)'
            book_chapter_cnt = 3
        elif series_name in ['twilight', 'hunger_games']:
            if day == 'at the end of the scene':
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][f'end of book{book_no}']
            else:
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][day]
            book_chapter_name = f'book number and chapter number'
            book_chapter_format = f'book M - chapter N (write Arabic number instead of book title for the book number)'
            book_chapter_cnt = 2
        else:
            raise ValueError
        series_full_name = series_name_dict[series_name]

        retrieved_docs = self.get_relevant_documents(series_name, question)
        filtered_docs = self.cutoff_docs(retrieved_docs,
                                         book_chapter_cnt,
                                         character_book_chapter_num_abs)

        if use_rag_for_temporal_expert:
            user_prompt = f"""You will be given a question and contexts from {series_full_name} series at a specific time. Your task is to identify the exact {book_chapter_name} of the scene of the question. Below is the data:
***
[Question]
{question}
***
[Contexts]
{self.format_docs(retrieved_docs, series_name, book_chapter_cnt)}
***
[Identification Criterion]
What is the exact {book_chapter_name} of the scene of the question?

[Identification Steps]
1. Read through the [Question] and [Contexts], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question, in '{book_chapter_format}' format.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""
        else:
            user_prompt = f"""You will be given a question from {series_full_name} series at a specific time. Your task is to identify the exact {book_chapter_name} of the scene of the question. Below is the data:
***
[Question]
{question}
***
[Identification Criterion]
What is the exact {book_chapter_name} of the scene of the question?

[Identification Steps]
1. Read through the [Question], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question, in '{book_chapter_format}' format.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""

        if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
            completion = call_openai_api(model_name, "You are a helpful and accurate assistant.", f"{user_prompt}",
                                            max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
            content = completion.choices[0].message.content
            print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
            print(f"# of completion tokens = {completion.usage.completion_tokens}")
        elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
            assert tokenizer is not None and model is not None and data_type is not None
            if model_name in ['llama-2-13b-chat']:
                messages = [
                    {"role": "system", "content": "You are a helpful and accurate assistant."},
                    {"role": "user", "content": user_prompt},
                ]
            elif model_name in ['mistral-instruct-7b']:
                messages = [
                    {"role": "user", "content": user_prompt},
                ]
            else:
                raise NotImplementedError
            content = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
        else:
            NotImplementedError
        rprint(f"[purple]Prompt: {user_prompt}[/purple]\n***\n[bold purple]Response: {content}\n***[/bold purple]")
        rprint(f"[bold purple]Gold question period: {question_period}[/bold purple]")

        temporal_label_str = extract_and_format_number(content.strip().split('\n')[-1].lower(), book_chapter_cnt)

        is_future = False
        if temporal_label_str == '':
            pass
        else:
            temporal_label = compare_book_chapters(temporal_label_str, character_book_chapter_num_abs)
            if temporal_label == 'after':
                hint.append(f"Note that the period of the question is in the future relative to {character}'s time point. Therefore, you should not answer the question or mention any facts that occurred after {character}'s time point.")
                is_future = True
            elif temporal_label == 'before':
                # hint.append(f"Note that the period of the question is in the past relative to {character}'s time point. Therefore, you can respond based on your past experience.")
                pass
            else:
                raise ValueError

        if is_future:
            pass
        else:
            if use_rag_for_spatial_expert:
                user_prompt = f"""You will be given a question, a character, and contexts from {series_full_name} series. Your task is to classify whether the character is a participant (i.e., present or absent) in the scene of the question. Below is the data:
***
[Question]
{question}
[Character]
{character}
***
[Contexts]
{self.format_docs(retrieved_docs, series_name, book_chapter_cnt)}
***
[Classification Criterion]
Is the character a participant in the scene of the question?

[Classification Steps]
1. Read through the [Question] and [Contexts], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question.
3. Write a list of every character involved in the scene described in the question, including those not explicitly mentioned in the question but who were present in the scene.
4. Compare the list of participants to the character. Check if the list of participants contains the character.
5. If the list contains the character, classify it as 'present'. Otherwise, classify it as 'absent'.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""
            else:
                user_prompt = f"""You will be given a question and a character from {series_full_name} series. Your task is to classify whether the character is a participant (i.e., present or absent) in the scene of the question. Below is the data:
***
[Question]
{question}
[Character]
{character}
***
[Classification Criterion]
Is the character a participant in the scene of the question?

[Classification Steps]
1. Read through the [Question], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question.
3. Write a list of every character involved in the scene described in the question, including those not explicitly mentioned in the question but who were present in the scene.
4. Compare the list of participants to the character. Check if the list of participants contains the character.
5. If the list contains the character, classify it as 'present'. Otherwise, classify it as 'absent'.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""

            if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
                completion = call_openai_api(model_name, "You are a helpful and accurate assistant.", f"{user_prompt}",
                                                max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
                content = completion.choices[0].message.content
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of completion tokens = {completion.usage.completion_tokens}")
            elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
                assert tokenizer is not None and model is not None and data_type is not None
                if model_name in ['llama-2-13b-chat']:
                    messages = [
                        {"role": "system", "content": "You are a helpful and accurate assistant."},
                        {"role": "user", "content": user_prompt},
                    ]
                elif model_name in ['mistral-instruct-7b']:
                    messages = [
                        {"role": "user", "content": user_prompt},
                    ]
                else:
                    raise NotImplementedError
                content = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
            else:
                raise NotImplementedError
            rprint(f"[purple]Prompt: {user_prompt}[/purple]\n***\n[bold purple]Response: {content}[/bold purple]")
            rprint(f"[bold purple]Gold participants: {participants}[/bold purple]")

            spatial_label_str = content.strip().split('\n')[-1].lower()
            if 'present' in spatial_label_str and 'absent' in spatial_label_str:
                pass
            elif 'present' in spatial_label_str:
                # hint.append(f"Note that {character} had participated in the scene described in the question. Therefore, you should not imply that {character} was absent in the scene.")
                pass
            elif 'absent' in spatial_label_str:
                hint.append(f"Note that {character} had not participated in the scene described in the question. Therefore, you should not imply that {character} was present in the scene.")
            else:
                pass

        # Final response
        if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview'] + ['llama-2-13b-chat']:
            final_query = question if len(hint) == 0 else f"{question}\n(HINT: {' '.join(hint)})"
        elif model_name in ['gpt-3.5-turbo-1106']:
            final_query = question if len(hint) == 0 else f"(HINT: {' '.join(hint)})\n{question}"
        elif model_name == 'mistral-instruct-7b':
            final_query = question if len(hint) == 0 else f"Question: {question}\n(HINT: {' '.join(hint)})\nResponse:"
        else:
            raise NotImplementedError

        if is_future:
            # Final response
            if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
                completion = call_openai_api(model_name, system_prompt, final_query,
                                             max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
                response = completion.choices[0].message.content
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of completion tokens = {completion.usage.completion_tokens}")
            elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
                assert tokenizer is not None and model is not None and data_type is not None
                if model_name in ['llama-2-13b-chat']:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": final_query},
                    ]
                elif model_name in ['mistral-instruct-7b']:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n***\n{final_query}"},
                    ]
                else:
                    raise NotImplementedError
                response = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
            else:
                raise NotImplementedError
            rprint(f"[blue]Prompt: {system_prompt}[/blue]\n***\n[green]({data_type}) {final_query}[/green]\n\n[bold green]Final Response: {response}\n***[/bold green]")
        else:
            if len(filtered_docs) == 0:
                rag_prompt = f""
            else:
                formatted_docs = self.format_docs(filtered_docs, series_name, book_chapter_cnt)
                rag_prompt = f"Context: {self.format_docs(filtered_docs, series_name, book_chapter_cnt)}\n***\n"

            # Final response
            if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
                completion = call_openai_api(model_name, system_prompt, f"{rag_prompt}{final_query}",
                                             max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
                response = completion.choices[0].message.content
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of completion tokens = {completion.usage.completion_tokens}")
            elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
                assert tokenizer is not None and model is not None and data_type is not None
                if model_name in ['llama-2-13b-chat']:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{rag_prompt}{final_query}"},
                    ]
                elif model_name in ['mistral-instruct-7b']:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n***\n{rag_prompt}{final_query}"},
                    ]
                else:
                    raise NotImplementedError
                response = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
            else:
                raise NotImplementedError
            rprint(f"[blue]Prompt: {system_prompt}[/blue]\n***\n[purple]{rag_prompt}[/purple][green]({data_type}) {final_query}[/green]\n\n[bold green]Final Response: {response}\n***[/bold green]")

        return response, '\n'.join(hint)