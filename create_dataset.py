import os
import re
import json
import copy
import random
import openai
import argparse
from tqdm import tqdm
from collections import defaultdict

from utils import (
    book_no2name,
    series_name_dict,
    preprocess_generation,
    call_openai_api,
)
openai.api_key = os.getenv('OPENAI_API_KEY')
random.seed(1234)

def parse_input_to_dict(input_text):
    # Splitting the input text into parts
    parts = input_text.split("\n\n")

    # Extracting the unique fact
    unique_fact = parts[0].split(": ")[1]

    # Extracting the participants
    participants = parts[1].split("\n")[1:]  # Ignore the first line "Participants:"

    # Constructing the dictionary
    output_dict = {
        "Unique Fact": unique_fact,
        "Participants": participants
    }

    return output_dict

def extract_and_format_number(chapter, number_cnt=2):
    formatted_str = ""
    # Extracting numbers using regular expression
    numbers = re.findall(r'\d+', chapter)
    if len(numbers) == number_cnt:
        if number_cnt == 2:
            # Formatting as 'BookNumber-ChapterNumber'
            formatted_str = f'{numbers[0]}-{numbers[1]}'
        elif number_cnt == 3:
            formatted_str = f'{numbers[0]}-{numbers[1]}-{numbers[2]}'
        else:
            raise ValueError
    else:
        raise ValueError
    return formatted_str

def generate_question_from_template(event, character):
    if event[-1] == '.':
        event = event[:-1]

    common_question_templates = [
        "Tell me your feelings when {event}.",
        "Tell me your genuine feelings when {event}.",
        "Describe your feelings when {event}.",
        "Describe your honest feelings when {event}.",
        "Can you describe your experience when {event}?",
        "Can you describe your true experience when {event}?",
    ]

    if character == 'Harry Potter':
        pass
    elif character in ['Ronald Weasley', 'Hermione Granger'] + ['Frodo Baggins', 'Gandalf', 'Aragorn', 'Samwise Gamgee', 'Legolas'] + ['Bella Swan', 'Edward Cullen', 'Jacob Black'] + ['Katniss Everdeen', 'Peeta Mellark', 'Gale Hawthorne']:
        common_question_templates += [
            "Did you see the moment when {event}?",
            "Did you truly see the moment when {event}?",
            "What did you see as {event}?",
            "What did you actually see as {event}?",
            "What did you hear when {event}?",
            "What did you precisely hear when {event}?",
            "Were you at the moment when {event}?",
            "Were you really at the moment when {event}?",
            "Were you present as {event}?",
            "Were you indeed present as {event}?",
            "Is it true that you were at the moment when {event}?",
            "Is it right that you were at the moment when {event}?",
        ]
    else:
        raise ValueError
    selected_question_template = random.choice(common_question_templates)
    question = selected_question_template.format(event=event)
    return question

def sample_character_period(question_book_chapter,
                            character=None,
                            series_name='harry_potter',
                            mode='past',
                            character_period_dict=None):
    assert mode in ['past', 'future']

    if series_name == 'harry_potter':
        character_period = {
            "on the 1st of September": ['1-6', '2-5', '3-5', '4-11', '5-10', '6-7', '7-12'],
            "on Halloween": ['1-10', '2-8', '3-8', '4-16'],
            "on Christmas": ['1-12', '2-12', '3-11', '4-23', '5-23', '6-16', '7-19'],
            "At the end of the scene": ['1-17', '2-18', '3-22', '4-37', '5-38', '6-30', '7-36']
        }

        question_book, question_chapter = question_book_chapter.split('-')
        is_question_period_future = dict()

        # include halloween
        if int(question_book) in [1,2,3,4]:
            for k,v in character_period.items():
                character_chapter = v[int(question_book) - 1].split('-')[1]
                if int(character_chapter) >= int(question_chapter):
                    is_question_period_future[k] = False
                else:
                    is_question_period_future[k] = True
        # exclude halloween
        else:
            for k,v in character_period.items():
                if k == 'on Halloween':
                    continue
                character_chapter = v[int(question_book) - 1].split('-')[1]
                if int(character_chapter) >= int(question_chapter):
                    is_question_period_future[k] = False
                else:
                    is_question_period_future[k] = True

        # past
        if mode == 'past':
            qualified_character_days = [k for k,v in is_question_period_future.items() if not v]
            assert len(qualified_character_days) > 0
            selected_character_day = random.choice(qualified_character_days)
            selected_character_book = copy.deepcopy(question_book)
        # future
        elif mode == 'future':
            qualified_character_days = [k for k,v in is_question_period_future.items() if v]
            if len(qualified_character_days) > 0:
                selected_character_day = random.choice(qualified_character_days)
                selected_character_book = copy.deepcopy(question_book)
            # if none of them are qualified, then let's go to "N-1"th grade to generate future category
            elif int(question_book) > 1:
                selected_character_day = random.choice(list(character_period.keys()))
                selected_character_book = f"{int(question_book) - 1}"
            else:
                selected_character_day = ''
                selected_character_book = copy.deepcopy(question_book)
        else:
            raise ValueError

        return selected_character_book, selected_character_day
    elif series_name in ['the_lord_of_the_rings', 'twilight', 'hunger_games']:
        raise NotImplementedError
    else:
        raise NotImplementedError

def main():
    assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='gpt-4-1106-preview', choices=['gpt-4-1106-preview'])
    parser.add_argument("--series_name", default='harry_potter', choices=['harry_potter', 'the_lord_of_the_rings', 'twilight', 'hunger_games'])
    parser.add_argument("--create_mode", default='generate_fact_event_summary', choices=['generate_fact_event_summary', 'generate_fact_freeform_question', 'generate_fake_event_summary', 'generate_fake_freeform_question', 'create_single_turn_dataset', 'generate_gold_response'])
    parser.add_argument("--dataset_dir", default='data/', type=str, help="dataset directoy")
    parser.add_argument("--cache_dir", default='', help='cache directory for event participants')
    parser.add_argument("--output_fname", default='final_dataset.json', type=str, help="output file name")
    args = parser.parse_args()

    # Load requirements
    if args.series_name == 'harry_potter':
        with open(os.path.join(args.dataset_dir, 'en_train_set.json'), 'r') as fp:
            data = json.load(fp)
        with open('data/event_participants_extraction_prompt/harry_potter.txt') as fp:
            initial_prompt = fp.read()
    elif args.series_name in ['the_lord_of_the_rings', 'twilight', 'hunger_games',]:
        raise NotImplementedError
    else:
        raise ValueError
    
    # Generate event summary based on the scene
    if args.create_mode == 'generate_fact_event_summary':
        source_data = []
        source_data_cnt = 0
        if args.series_name == 'harry_potter':
            for _, value in tqdm(data.items(), ncols=120):
                speakers = value['speakers']
                scene = value['scene']
                position = value['position']
                dialogue = value['dialogue']

                # filter short scene
                if len(' '.join(scene)) < 350:
                    continue

                prompt = initial_prompt.format(
                    position=position,
                    speakers= ', '.join(speakers),
                    scene=' '.join(scene)
                )

                # extract event and participants info from scene
                completion = call_openai_api(args.model_name, "You are a helpful and accurate assistant.", prompt)
                content = completion.choices[0].message.content
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of completion tokens = {completion.usage.completion_tokens}")
                print(content)

                content_dict = parse_input_to_dict(content)

                example_dict = {
                    'position': position,
                    'initial_speakers': speakers,
                    'scene': scene,
                    'dialogue': dialogue,
                    'event': content_dict['Unique Fact'],
                    'participants': content_dict['Participants']
                }

                source_data.append(example_dict)
                source_data_cnt += 1

                # cache source data
                if source_data_cnt % 10 == 0:
                    os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
                    with open(os.path.join(args.cache_dir, args.series_name, 'generated_fact_event_summary_temp.json'), 'w') as fp:
                        json.dump(source_data, fp, indent=4)
                    print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name, 'generated_fact_event_summary_temp.json')}!!\n\n")

            os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
            with open(os.path.join(args.cache_dir, args.series_name, 'generated_fact_event_summary.json'), 'w') as fp:
                json.dump(source_data, fp, indent=4)

            print(f"\n\nSaved generated_fact_event_summary.json to {os.path.join(args.cache_dir, args.series_name, 'generated_fact_event_summary.json')}!!\n\n")
            print(f'Good Job Computer!')

        elif args.series_name in ['the_lord_of_the_rings', 'twilight', 'hunger_games',]:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif args.create_mode == 'generate_fact_freeform_question':
        
        assert os.path.exists(os.path.join(args.cache_dir, args.series_name,'generated_fact_event_summary.json'))
        with open(os.path.join(args.cache_dir, args.series_name,'generated_fact_event_summary.json'), 'r') as fp:
            source_data = json.load(fp)

        source_data_cnt = 0
        if args.series_name in ['harry_potter']:
            for example_dict in tqdm(source_data, ncols=120):
                # Generate 5w1h question based on the event summary and GPT-4
                base_question = f"You will be given an event summary from the {series_name_dict[args.series_name]} series. Paraphrase the event summary to (1) a single-sentence question among 5w1h questions and (2) the answer to the question that should be answerable from the given event summary. Don't use pronouns to indicate the event, but self-contain what event it is. Note that the question should identify the unique period of the story."
                first_shot_example = f"Event summary: Ron's broken wand caused the charm to backfire, erasing Lockhart's memory and causing a portion of the ceiling to cave in.\nQuestion: What caused Gilderoy Lockhart's memory loss and the partial collapse of the ceiling?\nAnswer: Gilderoy Lockhart's memory was erased and a portion of the ceiling caved in when Ron Weasley's broken wand caused a backfired charm in their second year at Hogwarts."
                second_shot_example = f"Event summary: Harry uncovered that it was Professor Quirrell who attempted to seize the Sorcerer's Stone, revealing that he was under the influence of Lord Voldemort, who existed parasitically on the reverse side of Quirrell's head.\nQuestione Who did Harry Potter find out was attempting to steal the Sorcerer's Stone and was possessed by Lord Voldemort during their encounter at Hogwarts, and where was Voldemort residing on the individual's body?\nAnswer: Harry Potter discovered that Professor Quirrell, with Lord Voldemort residing on the back of his head, was trying to steal the Sorcerer's Stone."
                event = example_dict['event']
                completion = call_openai_api(args.model_name,
                                             "You are a helpful and accurate assistant.",
                                             f'{base_question}\n"""\n{first_shot_example}\n"""\n{second_shot_example}\n"""\nEvent summary: {event}\n')
                content = completion.choices[0].message.content
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of completion tokens = {completion.usage.completion_tokens}")
                print(content)

                pattern = r"Question: (.*)(\n*)(\s*)Answer: (.*)"
                # Search for the pattern in the input string
                match = re.search(pattern, content)

                if match:
                    example_dict['wh_question'] = match.group(1).strip()
                    example_dict['answer_to_wh_question'] = match.group(4).strip()
                example_dict['wh_question_with_answer'] = content

                source_data_cnt += 1
                # cache source data
                if source_data_cnt % 10 == 0:
                    os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
                    with open(os.path.join(args.cache_dir, args.series_name, 'generated_fact_freeform_question_temp.json'), 'w') as fp:
                        json.dump(source_data, fp, indent=4)
                    print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name,'generated_fact_freeform_question_temp.json')}!!\n\n")

            os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
            with open(os.path.join(args.cache_dir, args.series_name, 'generated_fact_freeform_question.json'), 'w') as fp:
                json.dump(source_data, fp, indent=4)

            print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name,'generated_fact_freeform_question.json')}!!\n\n")
            print(f'Good Job Computer!')
            return
        elif args.series_name in ['the_lord_of_the_rings', 'twilight', 'hunger_games',]:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif args.create_mode == 'generate_fake_event_summary':
        assert os.path.exists(os.path.join(args.cache_dir, args.series_name,'generated_fact_freeform_question.json'))
        with open(os.path.join(args.cache_dir, args.series_name,'generated_fact_freeform_question.json'), 'r') as fp:
            source_data = json.load(fp)

        source_data_cnt = 0
        if args.series_name in ['harry_potter']:
            for example_dict in tqdm(source_data, ncols=120):
                # Generate fake event summary based on the event summary and GPT-4
                base_question_fake = f"You will be given an event summary from the {series_name_dict[args.series_name]} series. Generate the fake event summary that converts the true event summary to confuse readers using one of the six methods as follows.\n***\n" \
                    +"1. Change the character: Swap the character with another character.\n" \
                    +"- True: Harry tricked Malfoy into freeing Dobby by giving Malfoy one of his own socks, which he promptly threw away, and was caught by Dobby.\n" \
                    +"- Fake: Harry tricked Snape into freeing Dobby by giving Snape one of his own socks, which he promptly threw away, and was caught by Dobby.\n" \
                    +"2. Change the Key Object: Alter the object that is central to the event.\n" \
                    +"- True: Harry used his own sock to free Dobby.\n" \
                    +"- Fake: Harry used a spellbook to free Dobby.\n" \
                    +"3. Alter the Location: Change the setting where the event took place.\n" \
                    +"- True: The event took place in Malfoy Manor.\n" \
                    +"- Fake: The event took place in the Gryffindor common room.\n" \
                    +"4. Switch the Action: Change what was done to the object or the action taken by the character.\n" \
                    +"- True: Malfoy threw the sock away.\n" \
                    +"- Fake: Malfoy donated the sock to charity.\n" \
                    +"5. Introduce a Nonexistent Character or Object: Add someone or something that wasn't originally there.\n" \
                    +"- True: Harry and Malfoy were the main characters involved.\n" \
                    +"- Fake: Harry, Malfoy, and a ghost named Sir Pudding were involved in the exchange.\n" \
                    +"6. Change the Character’s Knowledge: Switch what the character knows or doesn't know.\n" \
                    +"- True: Harry knew the sock would free Dobby.\n" \
                    +"- Fake: Harry had no idea that the sock would free Dobby and thought it was just a useless gift.\n***\n"
                first_shot_example_fake = f"True event summary: Harry received a Nimbus 2000, a gift from Professor McGonagall.\nFake event summary: Harry received a Nimbus 2000, a gift from Professor Snape.\nMethod number: 1. Change the character"
                second_shot_example_fake = f"True event summary: Fred, George, and Ron rescued Harry from the Dursleys with the use of a Flying Ford Anglia.\nFake event summary: Fred, George, and Ron rescued Harry from Hogwarts with the use of a Flying Ford Anglia\nMethod number: 3. Alter the Location"
                event = example_dict['event']
                completion = call_openai_api(args.model_name,
                                             "You are a helpful and accurate assistant.",
                                             f"{base_question_fake}{first_shot_example_fake}\n***\n{second_shot_example_fake}\n***\nTrue event summary: {event}")
                content = completion.choices[0].message.content
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of completion tokens = {completion.usage.completion_tokens}")
                print(f"{content}")

                pattern = r"Fake event summary: (.*)(\n*)(\s*)Method number: (.*)"
                # Search for the pattern in the input string
                match = re.search(pattern, content)

                if match:
                    example_dict['fake_event'] = match.group(1).strip()
                    example_dict['fake_event_method_num'] = match.group(4).strip()
                example_dict['fake_event_with_method_num'] = content

                source_data_cnt += 1
                # cache source data
                if source_data_cnt % 10 == 0:
                    os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
                    with open(os.path.join(args.cache_dir, args.series_name,'generated_fake_event_summary_temp.json'), 'w') as fp:
                        json.dump(source_data, fp, indent=4)
                    print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name,'generated_fake_event_summary_temp.json')}!!\n\n")

            os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
            with open(os.path.join(args.cache_dir, args.series_name,'generated_fake_event_summary.json'), 'w') as fp:
                json.dump(source_data, fp, indent=4)
            print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name,'generated_fake_event_summary.json')}!!\n\n")
            print(f'Good Job Computer!')
            return
        elif args.series_name in ['the_lord_of_the_rings', 'twilight', 'hunger_games']:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif args.create_mode == 'generate_fake_freeform_question':
        assert os.path.exists(os.path.join(args.cache_dir, args.series_name,'generated_fake_event_summary.json'))
        with open(os.path.join(args.cache_dir, args.series_name,'generated_fake_event_summary.json'), 'r') as fp:
            source_data = json.load(fp)

        source_data_cnt = 0
        if args.series_name in ['harry_potter']:
            for example_dict in tqdm(source_data, ncols=120):
                # Generate 5w1h question based on the event summary and GPT-4
                base_question = f"You will be given two event summaries from the {series_name_dict[args.series_name]} series. One is a true event summary and the other is the fake event summary which is generated from the true event summary to confuse readers. Paraphrase the fake event summary to (1) a single-sentence fake question among 5w1h questions and (2) the true answer to the question that should be answerable from the given true event summary. Don't use pronouns to indicate the event, but self-contain what event it is. Note that the question should identify the unique period of the story."
                first_shot_example = f"True event summary: Fred, George, and Ron rescued Harry from the Dursleys with the use of a Flying Ford Anglia.\nFake event summary: Fred, George, and Ron rescued Harry from Hogwarts with the use of a Flying Ford Anglia\nFake question: How did Fred, George, and Ron rescue Harry from Hogwarts using a Flying Ford Anglia?\nTrue answer: Fred, George, and Ron did not rescue Harry from Hogwarts; they rescued him from the Dursleys' house using a Flying Ford Anglia."
                second_shot_example = f"True event summary: Harry received a Nimbus 2000, a gift from Professor McGonagall.\nFake event summary: Harry received a Nimbus 2000, a gift from Professor Snape.\nFake question: Why did Professor Snape give Harry a Nimbus 2000?\nTrue answer: Professor Snape did not give Harry a Nimbus 2000; it was a gift from Professor McGonagall."
                true_event = example_dict['event']
                fake_event = example_dict['fake_event']
                completion = call_openai_api(args.model_name,
                                             "You are a helpful and accurate assistant.",
                                             f"{base_question}\n***\n{first_shot_example}\n***\n{second_shot_example}\n***\nTrue event summary: {true_event}\nFake event summary: {fake_event}\n",)
                content = completion.choices[0].message.content
                print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
                print(f"# of completion tokens = {completion.usage.completion_tokens}")
                print(f"{content}")

                pattern = r"Fake question: (.*)(\n*)(\s*)True answer: (.*)"
                # Search for the pattern in the input string
                match = re.search(pattern, content)

                if match:
                    example_dict['fake_wh_question'] = match.group(1).strip()
                    example_dict['answer_to_fake_wh_question'] = match.group(4).strip()
                example_dict['fake_wh_question_with_answer'] = content

                source_data_cnt += 1
                # cache source data
                if source_data_cnt % 10 == 0:
                    os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
                    with open(os.path.join(args.cache_dir, args.series_name,'generated_fake_freeform_question_temp.json'), 'w') as fp:
                        json.dump(source_data, fp, indent=4)
                    print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name,'generated_fake_freeform_question_temp.json')}!!\n\n")

            os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
            with open(os.path.join(args.cache_dir, args.series_name,'generated_fake_freeform_question.json'), 'w') as fp:
                json.dump(source_data, fp, indent=4)

            print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name,'generated_fake_freeform_question.json')}!!\n\n")
            print(f'Good Job Computer!')
            return
        elif args.series_name in ['the_lord_of_the_rings', 'twilight', 'hunger_games',]:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif args.create_mode == 'create_single_turn_dataset':
        assert os.path.exists(os.path.join(args.cache_dir, args.series_name,'generated_fake_freeform_question.json'))
        with open(os.path.join(args.cache_dir, args.series_name,'generated_fake_freeform_question.json'), 'r') as fp:
            source_data = json.load(fp)

        source_data_cnt = 0
        if args.series_name == 'harry_potter':
            book2year = {
                '1': '1st-year',
                '2': '2nd-year',
                '3': '3rd-year',
                '4': '4th-year',
                '5': '5th-year',
                '6': '6th-year',
                '7': '7th-year'
            }

            fact_structured_data_past = []
            fact_structured_data_future= defaultdict(list)
            fact_freeform_data = defaultdict(list)
            fake_freeform_data = defaultdict(list)
            characters = ['Hermione Granger', 'Ronald Weasley', 'Harry Potter']
            for example in tqdm(source_data, ncols=120):
                question_book_chapter = extract_and_format_number(example['position'])

                # check participants (should be filtered while creating source_data_filtered.json)
                for character in characters:
                    is_participant = any(character.lower() in participant.lower() for participant in example['participants'])

                    for mode in ['past', 'future']:
                        question = generate_question_from_template(example['event'], character)
                        book, day = sample_character_period(question_book_chapter,
                                                            character=character,
                                                            series_name=args.series_name,
                                                            mode=mode)
                        year = book2year[book]
                        book_name = book_no2name[args.series_name][book]

                        # future
                        if mode == 'future':
                            if day == '':
                                continue

                            if day == 'At the end of the scene':
                                is_end_scene = True
                                if book == '7':
                                    prefix = f"At the end of the scene of {book_name}"
                                else:
                                    prefix = f"At the end of the scene of {book_name} as a {year} student"
                            else:
                                is_end_scene = False

                            if character == 'Hermione Granger':
                                if is_end_scene:
                                    temporal_label = f"Future: {prefix}, Hermione Granger should (1) not be aware of or (2) contain any expression that reveals the moment when {example['event']} (Since the moment is the future for her.)"
                                else:
                                    temporal_label = f"Future: During her {year} {day}, Hermione Granger should (1) not be aware of or (2) contain any expression that reveals the moment when {example['event']} (Since the moment is the future for her.)"
                            elif character in ['Ronald Weasley', 'Harry Potter']:
                                if is_end_scene:
                                    temporal_label = f"Future: {prefix}, {character} should (1) not be aware of or (2) contain any expression that reveals the moment when {example['event']} (Since the moment is the future for him.)"
                                else:
                                    temporal_label = f"Future: During his {year} {day}, {character} should (1) not be aware of or (2) contain any expression that reveals the moment when {example['event']} (Since the moment is the future for him.)"
                            else:
                                raise ValueError
                            spatial_label = '-'
                            data_type = 'future'
                        # past
                        else:
                            assert day != ''
                            if day == 'At the end of the scene':
                                is_end_scene = True
                                if book == '7':
                                    prefix = f"At the end of the scene of {book_name}"
                                else:
                                    prefix = f"At the end of the scene of {book_name} as a {year} student"
                            else:
                                is_end_scene = False

                            # past
                            if character == 'Hermione Granger':
                                if is_end_scene:
                                    temporal_label = f"Past: {prefix}, Hermione Granger can respond based on the moment but should not wrongly recall it." + \
                                        "\n" + f"Moment (position: {example['position']}, speakers: {', '.join(example['initial_speakers'])}): " + ' '.join(example['scene'])
                                else:
                                    temporal_label = f"Past: During her {year} {day}, Hermione Granger can respond based on the moment but should not wrongly recall it." + \
                                        "\n" + f"Moment (position: {example['position']}, speakers: {', '.join(example['initial_speakers'])}): " + ' '.join(example['scene'])
                            elif character in ['Ronald Weasley', 'Harry Potter']:
                                if is_end_scene:
                                    temporal_label = f"Past: {prefix}, {character} can respond based on the moment but should not wrongly recall it." + \
                                        "\n" + f"Moment (position: {example['position']}, speakers: {', '.join(example['initial_speakers'])}): " + ' '.join(example['scene'])
                                else:
                                    temporal_label = f"Past: During his {year} {day}, {character} can respond based on the moment but should not wrongly recall it." + \
                                        "\n" + f"Moment (position: {example['position']}, speakers: {', '.join(example['initial_speakers'])}): " + ' '.join(example['scene'])
                            else:
                                raise ValueError

                            # spatial label
                            if is_participant:
                                opposite_status_adjective = 'absent'
                                status_noun = 'Presence'
                                data_type = 'past-presence'
                            else:
                                opposite_status_adjective = 'present'
                                status_noun = 'Absence'
                                data_type = 'past-absence'
                            if character == 'Hermione Granger':
                                if is_end_scene:
                                    spatial_label = f"{status_noun}: {prefix}, Hermione Granger should not say that she was {opposite_status_adjective} when {example['event']}"
                                else:
                                    spatial_label = f"{status_noun}: During her {year} {day}, Hermione Granger should not say that she was {opposite_status_adjective} when {example['event']}"
                            elif character in ['Ronald Weasley', 'Harry Potter']:
                                if is_end_scene:
                                    spatial_label = f"{status_noun}: {prefix}, {character} should not say that he was {opposite_status_adjective} when {example['event']}"
                                else:
                                    spatial_label = f"{status_noun}: During his {year} {day}, {character} should not say that he was {opposite_status_adjective} when {example['event']}"
                            else:
                                raise ValueError

                        if data_type == "future":
                            fact_structured_data_future[f"[future] {example['event']}"].append({
                                'series': args.series_name,
                                'fake_method': '-',
                                'question_generation': 'fact_structured',
                                'event_summary': example['event'],
                                'question_period': example['position'],
                                'question': question,
                                'character': character,
                                'character_period': f"{year}" if is_end_scene else f"{year} / {day}",
                                'participants': example['participants'],
                                'data_type': data_type,
                                'temporal_label': temporal_label,
                                'spatial_label': spatial_label,
                            })
                        else:
                            fact_structured_data_past.append({
                                'series': args.series_name,
                                'fake_method': '-',
                                'question_generation': 'fact_structured',
                                'event_summary': example['event'],
                                'question': question,
                                'question_period': example['position'],
                                'character': character,
                                'character_period': f"{year}" if is_end_scene else f"{year} / {day}",
                                'participants': example['participants'],
                                'data_type': data_type,
                                'temporal_label': temporal_label,
                                'spatial_label': spatial_label,
                            })

                        freeform_sc = "future" if data_type == "future" else "past-only"
                        if 'wh_question' in example.keys() and 'answer_to_wh_question' in example.keys():
                            fact_freeform_data[f"[{freeform_sc}] {example['event']}"].append({
                                'series': args.series_name,
                                'fake_method': '-',
                                'question_generation': 'fact_freeform',
                                'event_summary': example['event'],
                                'question': example['wh_question'],
                                'question_period': example['position'],
                                'character': character,
                                'character_period': f"{year}" if is_end_scene else f"{year} / {day}",
                                'participants': example['participants'],
                                'data_type': freeform_sc,
                                'temporal_label': temporal_label if data_type == "future" else f"{temporal_label}\nAnswer: {example['answer_to_wh_question']}",
                                'spatial_label': '-',
                            })

                        fake_event_sc = "future" if data_type == "future" else "past-only"
                        if fake_event_sc == 'past' and 'fake_wh_question' in example.keys() and 'answer_to_fake_wh_question' in example.keys():
                            fake_freeform_data[f"[{fake_event_sc}] {example['event']}"].append({
                                'series': args.series_name,
                                'fake_method': example['fake_event_method_num'],
                                'question_generation': 'fake_freeform',
                                'event': example['event'],
                                'question': example['fake_wh_question'],
                                'question_period': example['position'],
                                'character': character,
                                'character_period': f"{year}" if is_end_scene else f"{year} / {day}",
                                'participants': example['participants'],
                                'data_type': fake_event_sc,
                                'temporal_label': f"{temporal_label}\nAnswer: {example['answer_to_fake_wh_question']}",
                                'spatial_label': '-',
                            })
            sampled_fact_structured_data_future = [random.choice(v) for k,v in fact_structured_data_future.items()]
            sampled_fact_freeform_data = [random.choice(v) for k,v in fact_freeform_data.items()]
            sampled_fake_freeform_data = [random.choice(v) for k,v in fake_freeform_data.items()]
            final_data = fact_structured_data_past + sampled_fact_structured_data_future + sampled_fact_freeform_data + sampled_fake_freeform_data

            os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
            with open(os.path.join(args.cache_dir, args.series_name,'generated_single_turn_dataset.json'), 'w') as fp:
                json.dump(final_data, fp, indent=4)

            print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name,'generated_single_turn_dataset.json')}!!\n\n")
            print(f'Good Job Computer!')
            return
        elif args.series_name in ['the_lord_of_the_rings', 'twilight', 'hunger_games']:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif args.create_mode == 'generate_gold_response':
        assert os.path.exists(os.path.join(args.cache_dir, args.series_name,'generated_single_turn_dataset.json'))
        with open(os.path.join(args.cache_dir, args.series_name,'generated_single_turn_dataset.json'), 'r') as fp:
            source_data = json.load(fp)
        outputs = []

        data_cnt = 0
        for example in tqdm(source_data, ncols=120):
            series = example['series']
            data_type = example['data_type']
            character = example['character']
            character_period = example['character_period']
            question = example['question']

            system_prompt, _, _, _ = preprocess_generation(series, character, character_period)

            if data_type in ['future', 'past-only']:
                rationale = example['temporal_label']
            elif data_type in ['past-absence', 'past-presence']:
                rationale = example['spatial_label']
            else:
                raise ValueError
            
            prompt = f"Given a [Question] and the [Rationale] behind its answer, reply to the [Question] in accordance with the answer's [Rationale], while acting like {character}."
            prompt += f"\n***\n[Question]: {question}\n***\n[Rationale]:\n{rationale}\n***\nResponse: "

            completion = call_openai_api(args.model_name, system_prompt, prompt,
                                         max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
            response = completion.choices[0].message.content
            print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
            print(f"# of completion tokens = {completion.usage.completion_tokens}")
            print(f"Prompt: {system_prompt}\n{prompt}\n\n[{data_type}]Question: {question}\n\nResponse: {response}")

            output = copy.deepcopy(example)
            output['gold_response'] = copy.deepcopy(response)
            outputs.append(output)
            data_cnt += 1

            # save temporary outputs
            if data_cnt % 10 == 0:
                os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
                with open(os.path.join(args.cache_dir, args.series_name, f'{args.output_fname}.temp'), 'w') as fp:
                    json.dump(outputs, fp, indent=4)
                print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name, f'{args.output_fname}.temp')}!!\n\n")
        
        # save outputs
        os.makedirs(os.path.join(args.cache_dir, args.series_name), exist_ok=True)
        with open(os.path.join(args.cache_dir, args.series_name, args.output_fname), 'w') as fp:
            json.dump(outputs, fp, indent=4)
        print(f"\n\nSaved outputs to {os.path.join(args.cache_dir, args.series_name, args.output_fname)}!!\n\n")

        print('Good Job Computer!')
        return
    else:
        raise ValueError
if __name__ == '__main__':
    main()