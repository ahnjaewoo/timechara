import re
import time
import torch
import openai

book_no2name =  {
    'harry_potter': {
        '1': "Harry Potter and the Philosopher's Stone",
        '2': "Harry Potter and the Chamber of Secrets",
        '3': "Harry Potter and the Prisoner of Azkaban",
        '4': "Harry Potter and the Goblet of Fire",
        '5': "Harry Potter and the Order of the Phoenix",
        '6': "Harry Potter and the Half-Blood Prince",
        '7': "Harry Potter and the Deathly Hallows"
    },
    'the_lord_of_the_rings': {
        '1': "The Fellowship of the Ring",
        '2': "The Two Towers",
        '3': "The Return of the King",
    },
    'twilight': {
        '1': "Twilight",
        '2': "New Moon",
        '3': "Eclipse",
        '4': "Breaking Dawn",
    },
    'hunger_games': {
        '1': "The Hunger Games",
        '2': "Catching Fire",
        '3': "Mocking Jay",
    }
}

series_name_dict = {
    'harry_potter': 'Harry Potter',
    'the_lord_of_the_rings': 'The Lord of The Rings',
    'twilight': 'Twilight',
    'hunger_games': 'The Hunger Games',
}

series2source = {
    'harry_potter': "from J.K. Rowling's Harry Potter novel series",
    'the_lord_of_the_rings': "from J.R.R. Tolkien's The Lord of the Rings novel series",
    'twilight': "from Stephenie Meyer's Twilight novel series",
    'hunger_games': "from Suzanne Collins's The Hunger Games novel series"
}

character_dict = {'Hermione Granger': 'hermione_granger', 'Ronald Weasley': 'ronald_weasley', 'Harry Potter': 'harry_potter',
                  'Frodo Baggins': 'frodo_baggins', 'Gandalf': 'gandalf', 'Aragorn': 'aragorn', 'Samwise Gamgee': 'samwise_gamgee', 'Legolas': 'legolas',
                  'Bella Swan': 'bella_swan', 'Edward Cullen': 'edward_cullen', 'Jacob Black': 'jacob_black',
                  'Katniss Everdeen': 'katniss_everdeen', 'Peeta Mellark': 'peeta_mellark', 'Gale Hawthorne': 'gale_hawthorne',}

character_period_harry_potter = {
    "on the 1st of september": ['1-6', '2-5', '3-5', '4-11', '5-10', '6-7', '7-12'],
    "on halloween": ['1-10', '2-8', '3-8', '4-16'],
    "on christmas": ['1-12', '2-12', '3-11', '4-23', '5-23', '6-16', '7-19'],
    "at the end of the scene": ['1-17', '2-18', '3-22', '4-37', '5-38', '6-30', '7-36']
}

character_period_the_lord_of_the_rings = {
    'Frodo Baggins': {
        "at Bilbo Baggins's Farewell party": '1-1-1',
        'at the moment when Frodo was stabbed by one of the Ringwraiths': '1-1-11',
        'at the moment when The Fellowship loses Gandalf in Moria': '1-2-5',
        'end of Volume1': '1-2-10',
        'at the moment when encountering Gollum and decides to spare his life': '2-4-1',
        'at the moment of encountering the Black Gate of Mordor': '2-4-3',
        'at the moment when Frodo captured by Faramir': '2-4-5',
        "at Shelob's lair": '2-4-9',
        'end of Volume2': '2-4-10',
        'at the event when captured by Orcs at the Tower of Cirith Ungol': '3-6-1',
        'at the moment when rescued from Mount Doom by the eagles': '3-6-4',
        'end of Volume3': '3-6-9'
    },
    'Gandalf': {
        "at Bilbo Baggins's Farewell party": '1-1-1',
        'at the moment when the Fellowship was formed at the council of Elrond': '1-2-3',
        'at the moment when The Fellowship loses Gandalf in Moria': '1-2-5',
        'end of Volume1': '1-2-10',
        'at the moment when Gandalf the white met Aragorn, Legolas, and Gimli at Fangorn Forest after the fall at Moria': '2-3-5',
        "at the moment when arriving at Isengard after the battle of Helm's Deep": '2-3-8',
        'at the moment when Gandalf met Saruman at Isengard': '2-3-10',
        'end of Volume2': '2-4-10',
        'at the moment when Gandalf arrived at Minas Tirith with Pippin': '3-5-1',
        'during the Battle of the Pelennor Fields': '3-5-6',
        'at the coronation of King Elessar(Aragorn)': '3-6-5',
        'end of Volume3': '3-6-9'
    },
    'Aragorn': {
        'at the moment when Aragorn first met Frodo and his companions': '1-1-9',
        'at the moment when the Fellowship was formed at the council of Elrond': '1-2-3',
        'at the moment when The Fellowship loses Gandalf in Moria': '1-2-5',
        'end of Volume1': '1-2-10',
        'at the moment when Gandalf the white met Aragorn, Legolas, and Gimli at Fangorn Forest after the fall at Moria': '2-3-5',
        "at the moment when Aragorn arrived at Isengard after the battle of Helm's Deep": '2-3-8',
        'at the moment when Aragorn met Saruman at Isengard': '2-3-10',
        'end of Volume2': '2-4-10',
        'at the moment when Aragorn arrived at the paths of the dead': '3-5-2',
        'at the Battle of the Pelennor Fields': '3-5-6',
        'at the coronation of King Elessar(Aragorn)': '3-6-5',
        'end of Volume3': '3-6-9'
    },
    'Samwise Gamgee': {
        "at Bilbo Baggins's Farewell party": '1-1-1',
        'at the moment when Frodo was stabbed by one of the Ringwraiths': '1-1-11',
        'at the moment when The Fellowship loses Gandalf in Moria': '1-2-5',
        'end of Volume1': '1-2-10',
        'at the moment when encountering Gollum and decides to spare his life': '2-4-1',
        'at the moment of encountering the Black Gate of Mordor': '2-4-3',
        'at the moment when captured by Faramir': '2-4-5',
        "at Shelob's lair": '2-4-9',
        'end of Volume2': '2-4-10',
        'at the event when captured by Orcs at the Tower of Cirith Ungol': '3-6-1',
        'at the moment when rescued from Mount Doom by the eagles': '3-6-4',
        'end of Volume3': '3-6-9'
    },
    'Legolas': {
        'at the moment when the Fellowship was formed at the council of Elrond': '1-2-3',
        'at the moment when The Fellowship loses Gandalf in Moria': '1-2-5',
        'at the moment of leaving Lothlórien': '1-2-8',
        'end of Volume1': '1-2-10',
        "at the moment when Leoglas met Gandalf the white at Fangorn Forest after Gandalf's fall at Moria": '2-3-5',
        "at the moment when Legolas arrived at Isengard after the battle of Helm's Deep": '2-3-8',
        'at the moment when Legolas met Saruman at Isengard': '2-3-10',
        'end of Volume2': '2-4-10',
        'at the moment when Legolas arrived at the paths of the dead with Aragorn': '3-5-2',
        'at the Battle of the Pelennor Fields': '3-5-6',
        'at the coronation of King Elessar(Aragorn)': '3-6-5',
        'end of Volume3': '3-6-9'
    }
}


character_period_twilight = {
    'Bella Swan': {
        'at the moment when Bella moved from Phoenix to Forks': '1-1',
        "at the moment when Bella first confirm Edward's true nature as a vampire": '1-9',
        'at the moment when Bella first visited the Cullens': '1-15',
        'end of book1': '1-25',
        "on Bella's 18th birthday": '2-1',
        'at the moment when Bella jumps off the cliff into the ocean': '2-15',
        'at Volterra': '2-20',
        'end of book2': '2-25',
        'at the moment when Bella was grounded by her father': '3-1',
        'at the moment when Bella learns about the history of the Quileute tribe and the Cullens': '3-11',
        'at the moment when Bella receives an engagement ring from Edward': '3-20',
        'end of book3': '3-27',
        "at Bella and Edward's Wedding": '4-3',
        'at the moment when Renesmee was born': '4-18',
        'at the moment when Bella forges passports and IDs for Renesmee and Jacob from J. Jenks': '4-33',
        'end of book4': '4-39'
    },
    'Edward Cullen': {
        'at the moment when Edward saves Bella from a Van': '1-3',
        "at the moment when Bella first confirm Edward's true nature as a vampire": '1-9',
        'at the moment when Bella first visited the Cullens': '1-15',
        'end of book1': '1-25',
        "on Bella's 18th birthday": '2-1',
        'at the moment when Edward tells Bella that he and the Cullens are leaving Forks': '2-3',
        'at Volterra': '2-20',
        'end of book2': '2-25',
        'at the moment when Edward rewarded Alice for watching Bella by giving her the canary yellow Porsche from Italy': '3-6',
        "at Bella's graduation ceremony": '3-16',
        "at the moment when Jacob crawled into the sleeping bag beside Bella at the campsite, chosen for Bella's hiding place": '3-22',
        'end of book3': '3-27',
        "at Bella and Edward's Wedding": '4-3',
        'at the moment when Renesmee was born': '4-18',
        'at the moment when Esme has renovated a cottage on the property for Bella, Edward, and now Renesmee': '4-24',
        'end of book4': '4-39'
    },
    'Jacob Black': {
        'at the beach at La Push, when Jacob met Bella and her friends': '1-6',
        'on March 10, 2005': '1-11',
        'on March 13, 2005 when Bella found Jacob before watching a baseball game with Edward': '1-17',
        'end of book1': '1-25',
        'at the moment when Jacob and Bella worked together on repairing two old motorcycles': '2-6',
        "at the moment when Bella first discovers Jacob's werewolf identity": '2-10',
        'at the moment when Jacob pulled out Bella from drowning': '2-16',
        'end of book2': '2-25',
        'on May 31, 2006 when Bella found Jacob on his motorcycle at the school': '3-16',
        "at Bella's graduation ceremony": '3-7',
        "at the moment when Jacob crawled into the sleeping bag beside Bella at the campsite, chosen for Bella's hiding place": '3-22',
        'end of book3': '3-27',
        "at Bella and Edward's Wedding": '4-3',
        'at the moment when Renesmee was born': '4-18',
        'at Christmas, 2006': '4-34',
        'end of book4': '4-39'
    }
}

character_period_hunger_games = {
    "Katniss Everdeen": {
        "at the moment when Katniss volunteered to take her sister's place as the female tribute": "1-2",
        "at the start of the 74th Hunger Games": "1-11",
        "at the moment when Katniss found wounded Peeta hidden under a layer of mud": "1-19",
        "end of book1": "1-27",
        "at the moment when they arrived at District 11 for the first stop of the Victory Tour": "2-4",
        "at the announcement of the Quarter Quell": "2-12",
        "at the moment when Katniss first witnessed a heavy fog during the Quarter Quell": "2-20",
        "end of book2": "2-27",
        "at the first conversation about the bombing of district 12 with Gale": "3-1",
        "at the moment when Peeta suddenly warned of an impending attack on District 13": "3-9",
        "at the moment when the squad 451 was attacked by the mutts in the tunnels": "3-22",
        "end of book3": "3-28"
    },
    "Peeta Mellark": {
        "at the moment when Katniss volunteered to take her sister's place as the female tribute": "1-2",
        "at the start of the 74th Hunger Games": "1-11",
        "at the moment when Katniss found wounded Peeta hidden under a layer of mud": "1-19",
        "end of book1": "1-27",
        "at the moment when they arrived at District 11 for the first stop of the Victory Tour": "2-4",
        "at the announcement of the Quarter Quell": "2-12",
        "at the moment when Katniss first witnessed a heavy fog during the Quarter Quell": "2-20",
        "end of book2": "2-27",
        "at the moment when Peeta suddenly warned of an impending attack on District 13": "3-9",
        "at the moment when Peeta was sent as new member of squad 451 by president Coin": "3-18",
        "at the moment when the squad 451 was attacked by the mutts in the tunnels": "3-22",
        "end of book3": "3-28"
    },
    "Gale Hawthorne": {
        "at the moment when Katniss volunteered to take her sister's place as the female tribute": "1-2",
        "at the start of the 74th Hunger Games": "1-11",
        "at the moment when Katniss found wounded Peeta hidden under a layer of mud": "1-19",
        "end of book1": "1-27",
        "at the moment when Katniss delivered the animals she caught before the Victory Tour": "2-1",
        "at the announcement of the Quarter Quell": "2-12",
        "at the moment when Katniss first witnessed a heavy fog during the Quarter Quell": "2-20",
        "end of book2": "2-27",
        "at the first conversation about the bombing of district 12 with Katniss": "3-1",
        "at the moment when Peeta suddenly warned of an impending attack on District 13": "3-9",
        "at the moment when the squad 451 was attacked by the mutts in the tunnels": "3-22",
        "end of book3": "3-28"
    }
}

character_period = {
    'Harry Potter': character_period_harry_potter,
    'The Lord of The Rings': character_period_the_lord_of_the_rings,
    'Twilight': character_period_twilight,
    'The Hunger Games': character_period_hunger_games,
}

def preprocess_generation(series, character, character_period):
    source = series2source[series]
    if series == 'harry_potter':
        assert character in ['Hermione Granger', 'Ronald Weasley', 'Harry Potter']

        year = character_period.split('/')[0].strip()
        book_no = year[0]
        book_name = book_no2name[series][book_no]

        if len(character_period.split('/')) == 1:
            day = f'at the end of the scene'
            is_end_scene = True
        elif len(character_period.split('/')) == 2:
            day = character_period.split('/')[1].strip()
            is_end_scene = False
        else:
            raise ValueError

        is_end_of_the_series = is_end_scene and book_no == '7'
        if is_end_of_the_series:
            day_character = f'{character} {day} in {book_name}'
        else:
            day_character = f'{year} {character} {day} in {book_name}'

        if character == 'Hermione Granger':
            if is_end_of_the_series:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are at the end of the scene in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If she was not present at the location where the question was raised, she is likely unaware of the information or knowledge related to that question."
            elif is_end_scene:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are at the end of the scene in {book_name} as a {year} student and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If she was not present at the location where the question was raised, she is likely unaware of the information or knowledge related to that question."
            else:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are {day} during her {year} in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If she was not present at the location where the question was raised, she is likely unaware of the information or knowledge related to that question."
        elif character in ['Ronald Weasley', 'Harry Potter']:
            if is_end_of_the_series:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are at the end of the scene in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If he was not present at the location where the question was raised, he is likely unaware of the information or knowledge related to that question."
            elif is_end_scene:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are at the end of the scene in {book_name} as a {year} student and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If he was not present at the location where the question was raised, he is likely unaware of the information or knowledge related to that question."
            else:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are {day} during his {year} in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If he was not present at the location where the question was raised, he is likely unaware of the information or knowledge related to that question."
        else:
            raise ValueError

    elif series in ['the_lord_of_the_rings', 'twilight', 'hunger_games']:
        if series == 'the_lord_of_the_rings':
            assert character in ['Frodo Baggins', 'Gandalf', 'Aragorn', 'Samwise Gamgee', 'Legolas']
        elif series == 'twilight':
            assert character in ['Bella Swan', 'Edward Cullen', 'Jacob Black']
        elif series == 'hunger_games':
            assert character in ['Katniss Everdeen', 'Peeta Mellark', 'Gale Hawthorne']
        else:
            raise ValueError

        book_no = character_period.split('/')[0].strip()
        book_name = book_no2name[series][book_no]

        if len(character_period.split('/')) == 1:
            day = f'at the end of the scene'
            is_end_scene = True
        elif len(character_period.split('/')) == 2:
            day = character_period.split('/')[1].strip()
            is_end_scene = False
        else:
            raise ValueError

        day_character = f'{character} {day} in {book_name}'

        if character in ['Bella Swan'] + ['Katniss Everdeen']:
            if is_end_scene:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are at the end of the scene in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If she was not present at the location where the question was raised, she is likely unaware of the information or knowledge related to that question."
            else:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are {day} in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If she was not present at the location where the question was raised, she is likely unaware of the information or knowledge related to that question."
        elif character in ['Frodo Baggins', 'Gandalf', 'Aragorn', 'Samwise Gamgee', 'Legolas'] + ['Edward Cullen', 'Jacob Black'] + ['Peeta Mellark', 'Gale Hawthorne']:
            if is_end_scene:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are at the end of the scene in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If he was not present at the location where the question was raised, he is likely unaware of the information or knowledge related to that question."
            else:
                system_prompt = f"I want you to act like {character} {source}. I want you to respond and answer like {character}, using the tone, manner, and vocabulary {character} would use. Assume that you are {day} in {book_name} and interviewing with the interviewer. You should not answer the question and mention any fact that is future to the period. If he was not present at the location where the question was raised, he is likely unaware of the information or knowledge related to that question."
        else:
            raise ValueError
    else:
        raise NotImplementedError

    return system_prompt, book_no, day, day_character

def preprocess_evaluation(series, character, character_period):
    book_no = character_period.split('/')[0].strip()
    book_name = book_no2name[series][book_no[0]]
    if len(character_period.split('/')) == 1:
        day = f'at the end of the scene in {book_name}'
    elif len(character_period.split('/')) == 2:
        day = character_period.split('/')[1].strip()
    else:
        raise ValueError
    
    if series == 'harry_potter':
        assert character in ['Hermione Granger', 'Ronald Weasley', 'Harry Potter']
        if character_period.strip() == '7th-year':
            day_character = f"{character} {day}"
        else:
            day_character = f"{book_no} {character} {day}"
    elif series == 'the_lord_of_the_rings':
        assert character in ['Frodo Baggins', 'Gandalf', 'Aragorn', 'Samwise Gamgee', 'Legolas']
        day_character = f"{character} {day}"
    elif series == 'twilight':
        assert character in ['Bella Swan', 'Edward Cullen', 'Jacob Black']
        day_character = f"{character} {day}"
    elif series == 'hunger_games':
        assert character in ['Katniss Everdeen', 'Peeta Mellark', 'Gale Hawthorne']
        day_character = f"{character} {day}"
    else:
        raise NotImplementedError
    
    return day_character

def call_openai_api(model_name,
                    system_prompt,
                    question,
                    max_tokens=1024,
                    temperature=0.0,
                    top_p=0.95,
                    n=1,
                    seed=0):
    try:
        # Call the OpenAI API with your prompt
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            seed=seed,
        )
        return response
    except (RuntimeError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout) as e:
        print("Error: {}".format(e))
        time.sleep(2)
        return call_openai_api(model_name, system_prompt, question, max_tokens, temperature, top_p, n, seed)

def call_opensource_model(messages,
                          model,
                          tokenizer,
                          max_tokens=1024,
                          temperature=0.0,
                          top_p=0.95,
                          device='cpu'):
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device=device)
    input_length = input_ids.shape[1]
    attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device='cuda')
    generated_outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens)
    generated_ids = generated_outputs[0, input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response

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
            return formatted_str
    else:
        return formatted_str
    return formatted_str

def compare_book_chapters(chapter1, chapter2):
    # Split the strings by '-' and convert to integer lists
    list1 = list(map(int, chapter1.split('-')))
    list2 = list(map(int, chapter2.split('-')))

    # Compare element-wise
    for a, b in zip(list1, list2):
        if a < b:
            return "before"
        elif a > b:
            return "after"

    # If all elements are equal
    return "before"

def extract_score(content):
    last_sentence = content.strip().split('\n')[-1]
    matches = re.findall(r'\b[0-7]\b', last_sentence)
    if len(matches) == 0:
        score = content.strip()[-1]
        is_int = False
    else:
        score = int(matches[-1])
        is_int = True
    return score, is_int