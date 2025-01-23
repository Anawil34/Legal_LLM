# About
- This is one of my most challenging Hackathon in SuperAI Engineer SS4. The goal is to finetune LLM to be able to classify the legal activity whether it is legit or not.
- The data is Thai language, text-based, complex legal activity with many type of legal activites and people that performs said legal act.
- My team ultilize LLM finetuning techniques to create LLM model that accurately classify legal activaty with f1 score of 0.86 and can give easons for the said classification as well.

## Method
- Model used llama-3-8b-Instruct-bnb-4bit
- LlamaFactory to finetune the model with LORA + DPO

## Evaluation
| Model           | F1-Score |
|------------------|----------|
| Our Model  | 0.86    |

## Sample Input and Output
![image](https://github.com/user-attachments/assets/bb338a7c-b9dd-4af1-a204-b98f4c26246e)
![image](https://github.com/user-attachments/assets/b5b90a1a-7742-4169-b1b6-4fd5bbd62fb3)

![image](https://github.com/user-attachments/assets/2b6f9bd6-2a22-4e4e-867c-4bcdf11335f4)
![image](https://github.com/user-attachments/assets/07481b34-a544-41e2-a635-f459d6545b32)
