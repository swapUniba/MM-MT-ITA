from tqdm import tqdm
from together import Together

import os
import json
import time
import base64
import argparse


def main(model_id, ds_id, use_image, en_to_it):

    client = Together()

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
    
    if en_to_it:
        add_ += "_en_to_it"
    else:
        add_ += "_it_to_en"

    def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    def convert_to_conversation(sample, use_image, en_to_it):
                
        if en_to_it:

            if use_image:
                prompt = f"Translate the following text from English to Italian: \"{sample['lang_en']}\". Use the image as additional context for the translation. Provide only the translated text."
            else:
                prompt = f"Translate the following text from English to Italian: \"{sample['lang_en']}\". Provide only the translated text."
        else:

            if use_image:
                prompt = f"Translate the following text from Italian to English: \"{sample['lang_tgt']}\". Use the image as additional context for the translation. Provide only the translated text."
            else:
                prompt = f"Translate the following text from Italian to English: \"{sample['lang_tgt']}\". Provide only the translated text."
        
        if not os.path.isfile(sample["id"]):
            img_path = f"./babel_imgs/{sample['id']}.png"
        else:
            img_path = sample["id"]

        user_dict = { "role": "user",
            "content" : []
        }
            
        if use_image:
            encoded_img = encode_image(img_path)

            user_dict["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_img}",
                }}
            )
            
        user_dict["content"].append({
            "type": "text",
            "text": prompt
        })
        
        return user_dict
    
    os.makedirs(f"./outputs/{model_id}", exist_ok=True)

    with open(f"./outputs/{model_id}/{ds_id}{add_}_responses.jsonl", "w", encoding="utf8") as f: 

        for x in tqdm(data):

            messages = [convert_to_conversation(x, use_image, en_to_it)]

            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
                seed=42,
                temperature=0,
                n=1,
                top_k=1,
                top_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                max_tokens=128
            )

            response = ""

            for chunk in stream:

                try:
                    response += chunk.choices[0].delta.content
                except Exception as e:
                    response += ""
            
            x["generated_output"] = response

            json.dump(x, f)
            f.write('\n')

            time.sleep(5)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-d', '--ds_name')
    parser.add_argument('-i', '--use_image', action=argparse.BooleanOptionalAction)
    parser.add_argument('-ei', '--en_to_it', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    model_name = args.model_name
    ds_name = args.ds_name
    use_image = args.use_image
    en_to_it = args.en_to_it

    main(model_name, ds_name, use_image, en_to_it)
