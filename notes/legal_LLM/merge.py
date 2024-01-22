import json
import os

os.chdir(os.getcwd() + '/data/InterLM_format_data_ft')

def merge_jsonl(files, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in files:
            with open(fname, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

# JSONL 文件列表
jsonl_files = ['CrimeKgAssitant.jsonl', 'legal_article.jsonl', 'zh_contract_instruction.jsonl', 'zh_law_instruction.jsonl']  # , 'DISC-Law-SFT-Pair.jsonl'
output_file = 'law_merged2.jsonl'

merge_jsonl(jsonl_files, output_file)

