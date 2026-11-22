from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Any

from src.models import MinimalSearchResults, MinimalAnswer, Chunk
from src.error import RAGError

log = logging.getLogger(__name__)

QWEN3_CONTEXT = 32768
SAFETY_MARGIN = 64
SYSTEM_PROMPT = "Answer using only the provided context."


class AnswerGenerator:
    def __init__(
        self,
        name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 300,
        max_context_tokens: int = 6000,
    ) -> None:
        self.model_name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype="auto", device_map="auto"
        )
        self.max_generated_tokens = max_new_tokens
        self.budget = min(
            max_context_tokens, QWEN3_CONTEXT - max_new_tokens - SAFETY_MARGIN
        )

    def _token_count(self, text: str) -> int:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(ids)

    def _render_prompt(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\nQuestion:{query}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise RAGError("Chat template did not return a string.")
        return prompt

    def _fit_context(self, query: str, chunks: list[Chunk]) -> str:
        remaining = self.budget - self._token_count(
            self._render_prompt(query, "")
        )
        if remaining <= 0:
            raise RAGError("Question exceeds token budget.")
        parts: list[str] = []
        for rank, c in enumerate(chunks, 1):
            block = f"{c.file_path}:\n{c.text}\nend of source\n"
            cost = self._token_count(block)
            if cost > remaining:
                log.warning(
                    f"Context budget reached, dropped {len(chunks) - rank + 1}"
                    f" of {len(chunks)} sources."
                )
            parts.append(block)
            remaining -= cost
        return "".join(parts)

    def run(
        self, search_results: MinimalSearchResults, chunks: list[Chunk]
    ) -> MinimalAnswer:
        prompt = self._render_prompt(
            search_results.question,
            self._fit_context(search_results.question, chunks),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(
            **inputs, max_new_tokens=300, do_sample=False
        )
        answer_ids = output[0][inputs["input_ids"].shape[1]:]
        answer_result = self.tokenizer.decode(
            answer_ids, skip_special_tokens=True
        )
        if not isinstance(answer_result, str):
            raise RAGError("Decoder did not return a string.")

        return MinimalAnswer(
            question=search_results.question,
            question_id=search_results.question_id,
            retrieved_sources=search_results.retrieved_sources,
            answer=answer_result,
        )
