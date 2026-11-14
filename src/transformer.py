from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import MinimalSearchResults, MinimalAnswer, Chunk
# from src.prompt import PromptGenerator

class AnswerGenerator():

    def __init__(self, name: str = 'Qwen/Qwen3-0.6B') -> None:
        self.model_name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype='auto', device_map='auto')
    
    
    def _build_user_context(self, query: str, chunks: list[Chunk]) -> dict[str, str]:
        user_prompt = {'role': 'user', 'content': 'to_fill_in'}
        prompt = "Context: \n" + self._get_text_from_chunks(chunks)
        prompt += f'Question: \n {query}'
        user_prompt['content'] = prompt
        return user_prompt
    
    def _get_text_from_chunks(self, chunks: list[Chunk]) -> str:
        context = ""
        for c in chunks:
            context += c.file_path
            context += ':\n'
            context += c.text
            context += '\nend of source\n'
        return context

    def run(self, search_results: MinimalSearchResults, chunks: list[Chunk]) -> MinimalAnswer:
        messages = [{'role': 'system', 'content': 'answer using only the provided context'}]
        messages.append(self._build_user_context(search_results.question, chunks))

        prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                )
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)
        answer_ids = output[0][inputs['input_ids'].shape[1]:]
        answer_result = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        return MinimalAnswer(
                question=search_results.question,
                question_id=search_results.question_id,
                retrieved_sources=search_results.retrieved_sources,
                answer=answer_result
                )
        
