import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import sys
from huggingface_hub import login

from transformers import StoppingCriteria, StoppingCriteriaList

import json



# Define a custom stopping criteria class that stops when a specific token is generated
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the generated sequence ends with the stop_token_id
        return input_ids[0, -1] == self.stop_token_id

def phi_api_call(model_name, prompt, max_tokens):
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Define the stop token (e.g., the tokenizer's EOS token)
    stop_token_id = tokenizer.eos_token_id

    # Create a stopping criteria list with the custom criterion
    stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])

    # prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=max_tokens, return_dict_in_generate=True, output_scores=True, 
                             stopping_criteria=stopping_criteria)

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True).to("cpu")
    text = tokenizer.batch_decode(outputs.sequences)[0]
    #print(text)

    # Tokenize the input text and retrieve both tokens and the original text
    tokenized_output = tokenizer(text, return_offsets_mapping=True)

    # Get the tokens and their corresponding offsets
    offsets = tokenized_output['offset_mapping']


    # Writing Log file
    with open("log.txt", "w") as file:
        for tok, offset in zip(outputs.sequences[:, ][0], offsets):
            file.write(f"| {tok:5d} | {tokenizer.decode(tok):10s} | {offset[0]}\n")

    
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length: ][0]
    transition_scores = transition_scores[0][:]
    offsets = offsets[input_length: ]


    # Truncating Output
    for idx, tok in enumerate(generated_tokens):
        if tok == 198 and generated_tokens[idx + 1] == 18243 and generated_tokens[idx + 2] == 2482:
            generated_tokens = generated_tokens[:idx]
            transition_scores = transition_scores[:idx]
            offsets = offsets[:idx]
            break

    
    #for tok, score, offset in zip(generated_tokens, transition_scores, offsets):
        # | token | token string | log probability | probability
        #print(f"| {tok:5d} | {tokenizer.decode(tok):10s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%} | {offset[0]}")

    
    result = {
        "text": tokenizer.decode(generated_tokens),
        "logprobs": {
            "tokens": [tokenizer.decode(tok) for tok in generated_tokens],
            "token_logprobs": transition_scores.tolist(),
            "text_offset": [a[0] for a in offsets],
        },
        "finish_reason": "stop" if len(generated_tokens) < max_tokens else "length"
    }

    return {'model': model_name, 'choices': [result]}


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=64)
    args = parser.parse_args()

    print(args.prompt)
    """
    #model_name = "microsoft/phi-2"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    #model_name = "google/gemma-2-2b"
    # Get the multiline argument
    prompt = sys.argv[1]
    # Get the single word argument
    max_tokens = int(sys.argv[2])

    # Log in to Hugging Face with your access token
    
    #login(token=access_token)
    api_result = phi_api_call(model_name, prompt, max_tokens)
    with open('output.json', 'w') as file:
        # write variables using repr() function
        json.dump(api_result, file)

    
    
