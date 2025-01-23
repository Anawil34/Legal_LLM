from datasets import load_from_disk
import os
test_ds = load_from_disk('./data_hf/test/')
print(test_ds)

from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc

args = dict(
  model_name_or_path="llm_model/llama-3-8b-Instruct-bnb-4bit", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="lib/LLaMA-Factory/saves/unsloth-llama3-8b-2",            # load the saved LoRA adapters
  template="llama3",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
  use_unsloth=True,                     # use UnslothAI's LoRA optimization for 2x faster generation
)
chat_model = ChatModel(args)

print("-----------------make predict-----------------")

def predict(sample):
    query = sample["prompt"]
    messages = [{"role": "user", "content": query}]

    new_text = chat_model.chat(messages)
    return {"predict": new_text[0].response_text}

test_ds = test_ds.map(predict)
print("SAMPLE PREDICT: ")
print(test_ds[0])

print("-----------------map predict-----------------")

def map_predict(sample):
    if "YES" in sample["predict"]:
        return {"pred_map": 1}
    elif "NO" in sample["predict"] :
        return {"pred_map": 0}

test_ds = test_ds.map(map_predict)

print("SAMPLE MAPPED: ")
print(test_ds[0])


print("-----------------create submission-----------------")

import pandas as pd
submission = pd.read_csv('./sample_submission.csv')
submission.loc[3:, "answer"] = test_ds["pred_map"][3:]
submission["answer"] = submission["answer"].astype(int)
print(submission["answer"].value_counts())
submission.to_csv('./output_submission.csv', index=False)


print("-----------------DONE-----------------")