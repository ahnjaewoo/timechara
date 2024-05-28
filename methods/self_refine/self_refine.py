from rich import print as rprint
from utils import (
    call_openai_api,
    call_opensource_model,
    extract_score,
)

def call_feedback(model_name, feedback_prompt_template, agent_name, question, answer,
                  tokenizer=None, model=None, data_type=None, device='cpu'):
    prompt = feedback_prompt_template.format(agent_name=agent_name,
                                             question_0=question,
                                             response_0=answer)
    if model_name in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        completion = call_openai_api(model_name, 'You are a helpful and accurate assistant.', prompt)
        print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
        print(f"# of completion tokens = {completion.usage.completion_tokens}")
        content = completion.choices[0].message.content
    elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
        assert tokenizer is not None and model is not None and data_type is not None
        if model_name in ['llama-2-13b-chat']:
            messages = [
                {"role": "system", "content": "You are a helpful and accurate assistant."},
                {"role": "user", "content": prompt},
            ]
        elif model_name in ['mistral-instruct-7b']:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            raise NotImplementedError
        content = call_opensource_model(messages, model, tokenizer, max_tokens=1024, temperature=0.2, top_p=0.95, device=device)
    else:
        NotImplementedError
    rprint(f"[purple]Feedback:\n{content}[/purple]")
    return content

def call_refinement(model_name, system_prompt, agent_name, question, answers, feedbacks,
                    max_tokens, temperature, top_p, n, seed,
                    tokenizer=None, model=None, data_type=None, device='cpu'):
    assert len(answers) == len(feedbacks)
    single_turn_prompt_template = "***\n{agent_name}: {answer}\n\nFeedback: \n{feedback}\n"

    refine_prompt = f"Question: {question}\n"

    for answer,feedback in zip(answers, feedbacks):
        refine_prompt += single_turn_prompt_template.format(agent_name=agent_name,
                                                            question=question,
                                                            answer=answer,
                                                            feedback=feedback)

    if model_name in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        refine_prompt += f"***\nOkay, let's use this feedback to improve the response.\n\n{question}"
        completion = call_openai_api(model_name, system_prompt, refine_prompt,
                                     max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
        print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
        print(f"# of completion tokens = {completion.usage.completion_tokens}")
        content = completion.choices[0].message.content
    elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
        assert tokenizer is not None and model is not None and data_type is not None
        if model_name in ['llama-2-13b-chat']:
            refine_prompt += f"***\nOkay, let's use this feedback to improve the response.\n\n{question}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": refine_prompt},
            ]
        elif model_name in ['mistral-instruct-7b']:
            refine_prompt += f"***\nOkay, let's use this feedback to improve the response.\n\nQuestion: {question}\nResponse:"
            messages = [
                {"role": "user", "content": f"{system_prompt}\n***\n{refine_prompt}"},
            ]
        else:
            raise NotImplementedError
        content = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
    else:
        NotImplementedError
    rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\n[green]({data_type}) Question: {question}[/green]\n\n[bold green]Refined Response: {content}[/bold green]")
    return content

def self_refine(model_name,
                system_prompt,
                question,
                day_character,
                tokenizer=None,
                model=None,
                data_type=None,
                max_tokens=1024,
                temperature=0.0,
                top_p=0.95,
                n=1,
                seed=0,
                device='cpu'):
    # Define stopping criteria here
    def is_refinement_sufficient(is_score, score) -> bool:
        if is_score:
            if score >= 5:
                return True
            else:
                return False
        else:
            return False

    if model_name in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        completion = call_openai_api(model_name, system_prompt, question,
                                     max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
        response = completion.choices[0].message.content
        print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
        print(f"# of completion tokens = {completion.usage.completion_tokens}")
        with open('methods/self_refine/feedback.txt', 'r') as fp:
            feedback_prompt_template = fp.read()
    elif model_name in ['llama-2-13b-chat', 'mistral-instruct-7b']:
        assert tokenizer is not None and model is not None and data_type is not None
        if model_name in ['llama-2-13b-chat']:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        elif model_name in ['mistral-instruct-7b']:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n***\n{question}"},
            ]
        else:
            raise NotImplementedError
        response = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
        with open('methods/self_refine/feedback_open_source.txt', 'r') as fp:
            feedback_prompt_template = fp.read()
    else:
        raise NotImplementedError
    rprint(f"[blue]System Prompt: {system_prompt}[/blue]\n\n[green]({data_type}) Question: {question}[/green]\n\n[bold green]Initial Response: {response}[/bold green]")

    responses, feedbacks = [], []
    responses.append(response)
    for i in range(3):
        feedback = call_feedback(model_name, feedback_prompt_template, day_character, question, responses[-1],
                                 tokenizer, model, data_type, device=device)
        feedbacks.append(feedback)

        feedback_score, is_score = extract_score(feedback)

        if is_refinement_sufficient(is_score, feedback_score):
            break

        refined = call_refinement(model_name, system_prompt, day_character, question, responses, feedbacks,
                                  max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed,
                                  tokenizer=tokenizer, model=model, data_type=data_type, device=device)
        responses.append(refined)

    return responses[-1], feedbacks
