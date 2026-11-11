from transformers import AutoModelForCausalLLM, AutoTokenizer

# from src.prompt import PromptGenerator

class AnswerGenerator():

    def __init__(self, name: str = 'Qwen/Qwen3-0.6B') -> None:
        self.model_name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLLM.from_pretrained(name, torch_dtype='auto', device_map='auto')
        # self.prompt_gen = PromptGenerator()
 
