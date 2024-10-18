import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, StoppingCriteria
import logging


class DialogueGenerator:
    def __init__(self):
        self.dialogue_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.stopping_criteria = [DialogueStoppingCriteria(self.tokenizer)]

        self.sep_token1 = self.tokenizer.encode(",")[0]
        self.sep_token2 = self.tokenizer.encode("<|endoftext|>")[0]
        self.sep_token3 = self.tokenizer.encode("\n")[0]

        logging.getLogger('transformers').setLevel(logging.ERROR)


    def generate_dialogue(self, temp):
        model_input, input_length = self._get_model_input()

        while True:
            tokens = self._generate_tokens(model_input, temp)
            npc_response = tokens[:, input_length:]
            print(self.tokenizer.batch_decode(npc_response)[0])

            response = input("Your response ('exit' to quit): ")
            if response == "exit":
                break
        
            response_tokens = self.tokenizer.encode(response.strip())
            response_tokens.append(self.sep_token3)
            model_input = torch.cat((tokens, torch.tensor(response_tokens).unsqueeze(0)), 1)
            input_length = model_input.shape[1]
        

    def _generate_tokens(self, model_input, temp):
        return self.dialogue_model.generate(input_ids = model_input, 
                                            attention_mask=torch.ones(model_input.shape), 
                                            stopping_criteria = self.stopping_criteria, 
                                            temperature=temp, 
                                            do_sample=True,
                                            max_length = 1024)
    
    
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