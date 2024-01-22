import json
import os

def convert_format1(input_file, output_file):
    # Step 1: Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Step 2 & 3: Convert the format and prepare the output data
    output_data = []
    for entry in data:
        conversation_entry = {
            "conversation": [
                {
                    "system": "你是一位专业、经验丰富的法律专家。您总是根据输入的问题提供准确、全面和详细的答案",
                    "input": entry["input"],
                    "output": entry["output"]
                }
            ]
        }
        output_data.append(conversation_entry)

    # Step 4: Write the output data to a JSONL file
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in output_data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


def convert_format2(input_file, output_file):
    # Step 1: Read the input text file and process each line
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    output_data = []

    for line in lines:
        if line.strip():  # Ignore empty lines
            data = json.loads(line.strip())

            # Create a new formatted entry
            formatted_entry = {
                "conversation": [
                    {
                        "system": "你是一位专业、经验丰富的法律专家。您总是根据输入的问题提供准确、全面和详细的答案",
                        "input": data["input"],
                        "output": data["answer"]
                    }
                ]
            }

            output_data.append(formatted_entry)

    with open(output_file, 'w', encoding='utf-8') as file:
        for item in output_data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


def convert_format3(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    output_data = []

    for line in lines:
        data = json.loads(line.strip())
        formatted_entry = {
            "conversation": [
                {
                    "system": "你是一位专业、经验丰富的法律专家。您总是根据输入的问题提供准确、全面和详细的答案",
                    "input": data["input"],
                    "output": data["output"]
                }
            ]
        }
        output_data.append(formatted_entry)

    with open(output_file, 'w', encoding='utf-8') as file:
        for item in output_data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


def convert_format4(input_file, output_file):
    # Step 1: Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Step 2 & 3: Convert the format and prepare the output data
    output_data = []

    for item in data:
        if len(item["conversations"]) == 2:
            conversation_entry = {
                "conversation": [
                    {
                        "system": "你是一位专业、经验丰富的法律专家。您总是根据输入的问题提供准确、全面和详细的答案",
                        "input": item["conversations"][0]["value"],
                        "output": item["conversations"][1]["value"]
                    }
                ]
            }
            output_data.append(conversation_entry)

    # Step 4: Write the output data to a JSONL file
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in output_data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


current_dir = os.getcwd()
# 进入子目录/data
os.chdir(current_dir + '/data')

input_file_name1 = 'CrimeKgAssitant清洗后_52k.json'
input_file_name2 = 'legal_article/article.txt'
input_file_name3 = 'DISC-Law-SFT/DISC-Law-SFT-Pair.jsonl'
input_file_name4 = 'hanfei/data/zh_contract_instruction.json'
input_file_name6 = 'hanfei/data/zh_law_instruction.json'
# input_file_name5 = 'hanfei/data/zh_law_conversation.json'

output_file_name1 = 'InterLM_format_data_ft/CrimeKgAssitant.jsonl'
output_file_name2 = 'InterLM_format_data_ft/legal_article.jsonl'
output_file_name3 = 'InterLM_format_data_ft/DISC-Law-SFT-Pair.jsonl'
output_file_name4 = 'InterLM_format_data_ft/zh_contract_instruction.jsonl'
output_file_name6 = 'InterLM_format_data_ft/zh_law_instruction.jsonl'
# output_file_name5 = 'InterLM_format_data_ft/zh_law_conversation.jsonl'

convert_format1(input_file_name1, output_file_name1)
convert_format2(input_file_name2, output_file_name2)
convert_format3(input_file_name3, output_file_name3)
convert_format4(input_file_name4, output_file_name4)
convert_format4(input_file_name6, output_file_name6)
# convert_format4(input_file_name5, output_file_name5)

