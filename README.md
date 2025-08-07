# Code repository for "Is Multimodality still Required for Multimodal Machine Translation? A case study on English and Italian"

This repository contains the inference scripts and evaluation script for the multimodal machine translation case study from English to Italian and viceversa.

Before using the repository you should clone the data in the [data](./data) directory. You can find the data and the images at this [HF repo](https://huggingface.co/datasets/swap-uniba/MM-MT-ITA). After cloning, make sure to unzip the imgs.zip file.

To perform inference on Qwen VL, use the [inference_vl](./inference_vl.py) script, for Qwen use the [inference_text](./inference_text.py) script, for Qwen VL 72B and LLaMA Scout use the [inference_cloud](./inference_cloud.py) script.

For evaluation use the [eval_bm](./eval_bm.py) script, below there is a usage example:

```
python3 eval_bm.py -d ./outputs/outputs_local/outputs_test_set_multi/Qwen2.5-VL-7B-Instruct_no_image_beam_search_en_to_it.jsonl -l ./data/test_set_multi_no_img.jsonl
```

In the [requirements](./requirements) directory, you can find a copy of the libraries used for each step: "requirements_eval" are the requirements for evaluation, "requirements_inference_local" are the requirements for inference locally and "requirements_inference_cloud" are the requirements for inference on Qwen 2.5 72B and LLaMA 4 Scout.
