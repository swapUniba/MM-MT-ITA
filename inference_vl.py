import os
import json
import torch
import argparse

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from qwen_vl_utils import process_vision_info


def main(model_name, ds_name, use_image, en_to_it, beam_search):

    set_seed(42)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir="./cache"
    ).to("cuda:0").eval()
    
    processor = AutoProcessor.from_pretrained(model_name)

    data = []

    with open(f"./data/{ds_name}.jsonl", "r", encoding="utf8") as f:
        for l in f:
            print(l)
            line_data = json.loads(l)
            data.append(line_data)
    
    if use_image:
        add_ = "_with_image"
    else:
        add_ = "_no_image"
    
    if beam_search:
        add_ += "_beam_search"

    os.makedirs(f"./outputs_{ds_name}/", exist_ok=True)

    file_name = f"./outputs_{ds_name}/{model_name.split('/')[-1]}{add_}_en_to_it.jsonl" if en_to_it else f"./outputs_{ds_name}/{model_name.split('/')[-1]}{add_}_it_to_en.jsonl"
    with torch.no_grad():
        with open(file_name, "w", encoding="utf8") as f:

            for x in tqdm(data):

                if use_image:
                    
                    if en_to_it:
                        prompt = f"Translate the following text from English to Italian: \"{x['lang_en']}\". Use the image as additional context for the translation. Provide only the translated text."
                    else:
                        prompt = f"Translate the following text from Italian to English: \"{x['lang_tgt']}\". Use the image as additional context for the translation. Provide only the translated text."
                    
                    if not os.path.isfile(x["id"]):
                        img_path = f"./babel_imgs/{sample['id']}.png"
                    else:
                        img_path = x["id"]

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": img_path,
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                
                else:

                    if en_to_it:
                        prompt = f"Translate the following text from English to Italian: \"{x['lang_en']}\". Provide only the translated text."
                    else:
                        prompt = f"Translate the following text from Italian to English: \"{x['lang_tgt']}\". Provide only the translated text."

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                if use_image:
                    image_inputs = [y.resize((336, 336)) for y in image_inputs]
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                if beam_search:
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=3)
                else:
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=1)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                x["generated_output"] = output_text[0].replace("\"", "")
                json.dump(x, f)
                f.write('\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-d', '--ds_name')
    parser.add_argument('-i', '--use_image', action=argparse.BooleanOptionalAction)
    parser.add_argument('-ei', '--en_to_it', action=argparse.BooleanOptionalAction)
    parser.add_argument('-bm', '--beam_search', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    model_name = args.model_name
    ds_name = args.ds_name
    use_image = args.use_image
    en_to_it = args.en_to_it
    beam_search = args.beam_search

    main(model_name, ds_name, use_image, en_to_it, beam_search)