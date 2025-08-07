import json
import argparse

import evaluate

def main(ds_path, labels_path):

    meteor = evaluate.load('meteor')
    bleu = evaluate.load('bleu')
    comet = evaluate.load('comet')

    labels = []

    with open(labels_path, "r", encoding="utf8") as f:
        for l in f:
            line_data = json.loads(l)
            labels.append(line_data)

    data = []

    with open(ds_path, "r", encoding="utf8") as f:
        for l in f:
            line_data = json.loads(l)
            data.append(line_data)

    references = []
    predictions = []
    sources = []

    tgt_field = "lang_tgt" if "en_to_it" in ds_path else "lang_en"
    src_field = "lang_en" if "en_to_it" in ds_path else "lang_tgt"

    if "outputs_test_set_babel" in ds_path:
        mapping_babel = {}

        with open("./data/test_set_babel_eval.jsonl", "r", encoding="utf8") as f:
            for l in f:
                line_data = json.loads(l)

                mapping_babel[line_data["id"]] = line_data["it_lemmas"] if "en_to_it" in ds_path else line_data["en_lemmas"]

    for x, y in zip(data, labels):

        sources.append(y[src_field])
        references.append(y[tgt_field])
        predictions.append(x["generated_output"])

    meteor_scores = meteor.compute(references=references, predictions=predictions)["meteor"]
    bleu_scores = bleu.compute(references=[[x] for x in references], predictions=predictions)["bleu"]
    comet_scores = comet.compute(references=references, sources=sources, predictions=predictions)
    em_scores = sum([0 if ref.lower() != pred.lower() else 1 for ref, pred in zip(references, predictions)]) / len(references)

    if "outputs_test_set_babel" in ds_path:
        babel_mapping_score = 0

        for x in data:
            if x["generated_output"].lower().replace("\"", "") in mapping_babel[x["id"]]:
                babel_mapping_score += 1
        
        print("BABEL SCORE")
        print(babel_mapping_score / len(data))

    print("METEOR")
    print(meteor_scores)
    print("BLEU")
    print(bleu_scores)
    print("COMET")
    print(comet_scores["mean_score"])
    print("EM")
    print(em_scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ds_name')
    parser.add_argument('-l', '--labels_name')
    args = parser.parse_args()
    
    ds_name = args.ds_name
    labels_name = args.labels_name

    main(ds_name, labels_name)