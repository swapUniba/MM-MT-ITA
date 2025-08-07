import os
import json
import torch
import argparse

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def main(model_name, ds_name, en_to_it):

    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir="./cache"
    ).to("cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = []

    with open(f"./data/{ds_name}.jsonl", "r", encoding="utf8") as f:
        for l in f:
            print(l)
            line_data = json.loads(l)
            data.append(line_data)

    os.makedirs(f"./outputs_{ds_name}/", exist_ok=True)
    file_name = f"./outputs_{ds_name}/{model_name.split('/')[-1]}_en_to_it.jsonl" if en_to_it else f"./outputs_{ds_name}/{model_name.split('/')[-1]}_it_to_en.jsonl"
    with torch.no_grad():
        with open(file_name, "w", encoding="utf8") as f:

            for x in tqdm(data):
                
                if en_to_it:
                    prompt = f"Translate the following text from English to Italian: \"{x['lang_en']}\". Provide only the translated text."
                else:
                    prompt = f"Translate the following text from Italian to English: \"{x['lang_tgt']}\". Provide only the translated text."

                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=1024, do_sample=False, num_beams=1
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                x["generated_output"] = response
                json.dump(x, f)
                f.write('\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-d', '--ds_name')
    parser.add_argument('-ei', '--en_to_it', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    model_name = args.model_name
    ds_name = args.ds_name
    en_to_it = args.en_to_it

    main(model_name, ds_name, en_to_it)