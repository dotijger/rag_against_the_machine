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
    """Generate answers to questions from retrieved context.

    Attributes:
        model_name: Hugging Face ID of the model.
        tokenizer: The model's tokenizer.
        model: The loaded causal language model.
        max_generated_tokens: Maximum number of tokens in an answer.
        budget: Maximum number of prompt tokens, including the question
            and the chat template.
    """

    def __init__(
        self,
        name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 300,
        max_context_tokens: int = 6000,
    ) -> None:
        """Load the tokenizer and model and work out the prompt budget.

        The budget is max_context_tokens, capped so the prompt plus
        the answer always fits in Qwen3's context window.

        Args:
            name: Hugging Face model ID.
            max_new_tokens: Maximum number of tokens in an answer.
            max_context_tokens: Desired maximum prompt length in tokens.
        """
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
        """Return the number of tokens in text, without special tokens.

        Args:
            text: The text to measure.

        Returns:
            The token count.
        """
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(ids)

    def _render_prompt(self, query: str, context: str) -> str:
        """Build the full chat prompt for a question and its context.

        Qwen3's thinking mode is turned off so the model answers directly.

        Args:
            query: The question text.
            context: The source blocks to answer from.

        Returns:
            The prompt as a string, ending with the assistant turn opener.

        Raises:
            RAGError: If the chat template does not return a string.
        """
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
        """Pack as many chunks as fit in the budget into a context string.

        Chunks are added in rank order, each labelled with its file path.
        At the first chunk that does not fit, it and all lower-ranked
        chunks are dropped, and a warning says how many.

        Args:
            query: The question text, which also uses up budget.
            chunks: Retrieved chunks, best first.

        Returns:
            The context string; empty if not even the first chunk fits.

        Raises:
            RAGError: If the question and template alone exceed the
                budget.
        """
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
                break
            parts.append(block)
            remaining -= cost
        return "".join(parts)

    def run(
        self, search_results: MinimalSearchResults, chunks: list[Chunk]
    ) -> MinimalAnswer:
        """Answer a question from its retrieved chunks.

        Args:
            search_results: The question and its retrieved sources.
            chunks: The chunks for those sources, best first.

        Returns:
            The search result with the generated answer. All retrieved
            sources are listed, even ones dropped to fit the budget,
            a MinimalAnswer instance.

        Raises:
            RAGError: If the prompt cannot be built or decoding does not
                return a string.
        """
        prompt = self._render_prompt(
            search_results.question,
            self._fit_context(search_results.question, chunks),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(
            **inputs, max_new_tokens=self.max_generated_tokens, do_sample=False
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
