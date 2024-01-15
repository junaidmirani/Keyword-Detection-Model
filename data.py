

import re
import json


def parse_text_to_json(text):
    data = []
    current_qa_pair = {}

    for line in text.split('\n'):
        if line.startswith('Q:'):
            if current_qa_pair:
                data.append(current_qa_pair)
            current_qa_pair = {'question': line[3:].strip()}

    if current_qa_pair:
        data.append(current_qa_pair)

    return data


def convert_to_json_files(text_file_path, json_file_paths):
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()

    parsed_data = parse_text_to_json(text_content)

    for json_file_path in json_file_paths:
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(parsed_data, json_file, ensure_ascii=False, indent=2)


# Specify your input file path and output file paths as a list
text_file_path = 'NEW VISTE AI BOT Malaga FRESH copy.txt'
json_file_paths = ['train.json']

convert_to_json_files(text_file_path, json_file_paths)
