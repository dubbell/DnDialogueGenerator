import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, StoppingCriteria, GenerationConfig
import logging
import pickle
import torch.nn as nn


class DialogueGenerator:
    def __init__(self, model_name):
        logging.getLogger('transformers').setLevel(logging.ERROR)

        self.dialogue_model = pickle.load(open(model_name, "rb"))["model"]

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces = False)

        self.stopping_criteria = [DialogueStoppingCriteria(self.tokenizer)]

        self.sep_token1 = self.tokenizer.encode(",")[0]
        self.sep_token2 = self.tokenizer.encode("<|endoftext|>")[0]
        self.sep_token3 = self.tokenizer.encode("\n")[0]

        self.generation_config = GenerationConfig(repetition_penalty = 1.5, 
                                                  no_repeat_ngram_size = 5, 
                                                  length_penalty = 1.5,
                                                  num_beams = 5,
                                                  stop_strings = ["\n"])


    def generate_dialogue(self, temp):
        model_input, input_length = self._get_model_input()

        dialogue_line = 0

        while True:
            tokens = self._generate_tokens(model_input, temp)

            newline_indices = (tokens[0] == self.sep_token3).nonzero().squeeze()
            
            if self.sep_token3 not in tokens[0]:
                next_line_end = -1
            elif newline_indices.dim() == 0:
                next_line_end = newline_indices
                dialogue_line += 1
            else:
                next_line_end = newline_indices[dialogue_line]
                dialogue_line += 1

            npc_response = tokens[:, input_length:next_line_end]
            tokens = tokens[:, :next_line_end]


            # print("\n" + str(self.tokenizer.batch_decode(tokens.squeeze())) + "\n")

            print(f" NPC response: {self.tokenizer.batch_decode(npc_response)[0].strip()}")

            response = input("Your response: ")
            if response == "exit":
                break
        
            response_tokens = self.tokenizer.encode(response.strip())
            response_tokens.append(self.sep_token3)
            model_input = torch.cat((tokens, torch.tensor(response_tokens).unsqueeze(0)), 1)
            input_length = model_input.shape[1]
        

    def _generate_tokens(self, model_input, temp):
        return self.dialogue_model.generate(input_ids = model_input, 
                                            attention_mask=torch.ones(model_input.shape), 
                                            generation_config = self.generation_config,
                                            stopping_criteria = self.stopping_criteria, 
                                            temperature=temp, 
                                            do_sample=True,
                                            max_length = 1024,
                                            tokenizer = self.tokenizer)
    
    
    def _get_model_input(self):
        categories = input("Enter dialogue categories, separated by commas: ").split(",")
        personalities = input("Enter NPC personality traits, separated by commas: ").split(",")
        
        model_input = []

        for i, category in enumerate(categories):
            model_input.extend(self.tokenizer.encode(category.strip()))
            model_input.append(self.sep_token1 if i < len(categories) - 1 else self.sep_token2)

        for i, personality in enumerate(personalities):
            model_input.extend(self.tokenizer.encode(personality.strip()))
            model_input.append(self.sep_token1 if i < len(personalities) - 1 else self.sep_token2)
        
        return torch.tensor(model_input).unsqueeze(0), len(model_input)
    


class DialogueStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_token = tokenizer.encode("\n")[0]

    def __call__(self, input_ids, score, **kwargs):
        return input_ids[0, -1] == self.stop_token