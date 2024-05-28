import os
import time
import torch
from rich import print as rprint
from collections import defaultdict

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from utils import (
    series_name_dict,
    character_period,
    call_openai_api,
    call_opensource_model,
    extract_and_format_number,
    compare_book_chapters,
)

class RAGCutoff:
    def __init__(self,
                 openai_api_key,
                 rag_cache_dir):

        if os.path.exists(rag_cache_dir):
            self.vectorstore = {}
            for series_name in ['harry_potter', 'the_lord_of_the_rings', 'twilight', 'hunger_games']:
                self.vectorstore[series_name] = Chroma(
                    persist_directory=os.path.join(rag_cache_dir, series_name, "chroma.db"),
                    embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
                )
        else:
            raise NotImplementedError, f"Download Chroma DB files from 'https://drive.google.com/file/d/1ye55y2hE20tQES1Co1iI5Eq28xJ-WCFv/view?usp=sharing' to use RAGCutoff and NarrativeExpertsRAGCutoff!!"
            # self.vectorstore = {}
            # for series_name in ['harry_potter', 'the_lord_of_the_rings', 'twilight', 'hunger_games']:
            #     if series_name in ['harry_potter', 'twilight', 'hunger_games']:
            #         if series_name == 'harry_potter':
            #             max_book_num = 7
            #         elif series_name == 'twilight':
            #             max_book_num = 4
            #         elif series_name == 'hunger_games':
            #             max_book_num = 3
            #         loaders = []
            #         for book_no in range(1, max_book_num + 1):
            #             chapters_list = os.listdir(f'data/transcript/{series_name}/book_{book_no}')

            #             # Check if all elements are in the form of "chapter_N.txt"
            #             # and extract chapter numbers
            #             chapter_numbers = []
            #             all_in_correct_form = True
            #             for filename in chapters_list:
            #                 if filename.startswith("chapter_") and filename.endswith(".txt"):
            #                     try:
            #                         number = int(filename.replace("chapter_", "").replace(".txt", ""))
            #                         chapter_numbers.append(number)
            #                     except ValueError:
            #                         all_in_correct_form = False
            #                         break
            #                 else:
            #                     all_in_correct_form = False
            #                     break

            #             assert all_in_correct_form
            #             max_chapter_number = max(chapter_numbers)
            #             for chapter_no in range(1, max_chapter_number + 1):
            #                 assert os.path.exists(f'data/transcript/{series_name}/book_{book_no}/chapter_{chapter_no}.txt')
            #                 print(f'data/transcript/{series_name}/book_{book_no}/chapter_{chapter_no}.txt')
            #                 loaders.append(TextLoader(f'data/transcript/{series_name}/book_{book_no}/chapter_{chapter_no}.txt'))
            #         loader_all = MergedDataLoader(loaders=loaders)
            #     elif series_name == 'the_lord_of_the_rings':
            #         loaders = []
            #         for volume_no in range(1,4):
            #             chapters_list = os.listdir(f'data/transcript/the_lord_of_the_rings/volume_{volume_no}')

            #             # Parse the strings to extract book numbers and chapter numbers
            #             parsed_info = [(int(name.split('_')[1]), int(name.split('_')[3].split('.')[0])) for name in chapters_list]

            #             # Initialize a dictionary to hold the maximum chapter number for each book
            #             max_chapters = defaultdict(int)

            #             # Iterate through the parsed info to find the maximum chapter number for each book
            #             for book, chapter in parsed_info:
            #                 if chapter > max_chapters[book]:
            #                     max_chapters[book] = chapter

            #             # Convert the defaultdict to a regular dict for clearer presentation
            #             max_chapters_dict = dict(max_chapters)

            #             for book_no, max_chapter_number in max_chapters_dict.items():
            #                 for chapter_no in range(1, max_chapter_number + 1):
            #                     assert os.path.exists(f'data/transcript/the_lord_of_the_rings/volume_{volume_no}/book_{book_no}_chapter_{chapter_no}.txt')
            #                     print(f'data/transcript/the_lord_of_the_rings/volume_{volume_no}/book_{book_no}_chapter_{chapter_no}.txt')
            #                     loaders.append(TextLoader(f'data/transcript/the_lord_of_the_rings/volume_{volume_no}/book_{book_no}_chapter_{chapter_no}.txt'))
            #         loader_all = MergedDataLoader(loaders=loaders)
            #     else:
            #         raise ValueError
            #     docs_all = loader_all.load()

            #     text_splitter = RecursiveCharacterTextSplitter(
            #         chunk_size=1000, chunk_overlap=200, add_start_index=True
            #     )
            #     all_splits = text_splitter.split_documents(docs_all)
            #     # Recommend: change it to sentence-transformer (open-source embedding function)
            #     # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            #     # https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/chroma.py
            #     print(f'making vector store...')
            #     os.makedirs(os.path.join(rag_cache_dir, series_name), exist_ok=True)
            #     self.vectorstore[series_name] = Chroma.from_documents(
            #         documents=all_splits,
            #         embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
            #         persist_directory=os.path.join(os.path.join(rag_cache_dir, series_name), "chroma.db"),
            #     )
        # https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores.py
        self.retriever = {}
        for series_name in ['harry_potter', 'the_lord_of_the_rings', 'twilight', 'hunger_games']:
            self.retriever[series_name] = self.vectorstore[series_name].as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def format_docs(self, docs, series_name, book_chapter_count):
        new_docs = []
        for doc in docs:
            source = doc.metadata['source']
            context_book_chapter_num = extract_and_format_number(source, book_chapter_count)
            if series_name in ['harry_potter', 'twilight', 'hunger_games']:
                context_book_num, context_chapter_num = context_book_chapter_num.split('-')
                new_docs.append(f"[Book {context_book_num} - Chapter {context_chapter_num}]\n{doc.page_content}")
            elif series_name == 'the_lord_of_the_rings':
                context_volume_num, context_book_num, context_chapter_num = context_book_chapter_num.split('-')
                new_docs.append(f"[Volume {context_volume_num} - Book {context_book_num} - Chapter {context_chapter_num}]\n{doc.page_content}")
            else:
                raise ValueError
        return '\n\n'.join(new_docs)

    def cutoff_docs(self, docs, book_chapter_count, character_book_chapter_num):
        # Extract book number and chapter number from each path
        new_docs = []
        for doc in docs:
            source = doc.metadata['source']
            context_book_chapter_num = extract_and_format_number(source, book_chapter_count)
            output = compare_book_chapters(context_book_chapter_num, character_book_chapter_num)
            if output == 'before':
                new_docs.append(doc)
            elif output == 'after':
                continue
            else:
                raise ValueError
        return new_docs

    def get_relevant_documents(self, series_name, question):
        max_retries = 5  # Maximum number of retries
        retry_count = 0  # Current retry attempt
        while True:
            try:
                retrieved_docs = self.retriever[series_name].get_relevant_documents(question)
                break
            except Exception as e:
                print(f"Failed to retrieve relevant documents: {e}")
                retry_count += 1  # Increment the retry counter
                if retry_count >= max_retries:
                    print("Max retries reached, failing gracefully.")
                    retrieved_docs = []
                    break
                time.sleep(2)
        return retrieved_docs

    def generate(self,
                 model_name,
                 system_prompt,
                 question,
                 character,
                 series_name,
                 book_no,
                 day,
                 model=None,
                 tokenizer=None,
                 data_type=None,
                 max_tokens=1024,
                 temperature=0.0,
                 top_p=0.95,
                 n=1,
                 seed=0,
                 device='cpu'):
        if series_name == 'harry_potter':
            try:
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][day.lower()][int(book_no) - 1]
            except:
                character_book_chapter_num_abs = f'{book_no}-0'
            book_chapter_cnt = 2
        elif series_name == 'the_lord_of_the_rings':
            if day == 'at the end of the scene':
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][f'end of Volume{book_no}']
            else:
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][day]
            book_chapter_cnt = 3
        elif series_name in ['twilight', 'hunger_games']:
            if day == 'at the end of the scene':
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][f'end of book{book_no}']
            else:
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][character][day]
            book_chapter_cnt = 2
        else:
            raise ValueError

        retrieved_docs = self.get_relevant_documents(series_name, question)
        filtered_docs = self.cutoff_docs(retrieved_docs,
                                         book_chapter_cnt,
                                         character_book_chapter_num_abs)

        if len(filtered_docs) == 0:
            formatted_docs = ''
            prompt = f"{question}"
        else:
            formatted_docs = self.format_docs(filtered_docs, series_name, book_chapter_cnt)
            prompt = f"Context: {formatted_docs}\n***\n{question}"

        if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
            completion = call_openai_api(model_name, system_prompt, prompt,
                                         max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
            print(f"# of prompt tokens = {completion.usage.prompt_tokens}")
            print(f"# of completion tokens = {completion.usage.completion_tokens}")
            response = completion.choices[0].message.content
            rprint(f"[green]Prompt: {prompt} ({data_type})[/green]\n\n[bold green]Response: {response}[/bold green]")
        elif model_name in ['mistral-instruct-7b', 'llama-2-13b-chat']:
            assert tokenizer is not None and model is not None and data_type is not None
            if model_name in ['llama-2-13b-chat']:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            elif model_name in ['mistral-instruct-7b']:
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n***\n{prompt}\nResponse:"},
                ]
            else:
                raise NotImplementedError
            response = call_opensource_model(messages, model, tokenizer, max_tokens=max_tokens, temperature=temperature, top_p=top_p, device=device)
            rprint(f"[green]Prompt: {system_prompt}\n\n{prompt} ({data_type})[/green]\n\n[bold green]Response: {response}[/bold green]")
        else:
            raise NotImplementedError

        return response, formatted_docs