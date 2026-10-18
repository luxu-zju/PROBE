"""Prompt construction, text generation, and JSON output parsing."""

import json

from abc import ABC, abstractmethod
from typing import List

from source.pipeline.constants import DOC_DOC_DELIM, THOUGHT_THOUGHT_DELIM
from source.pipeline.state import AnswerState, BaseState, ThoughtState
from source.pipeline.utils import (
    clean_json_value,
    parse_path,
    retrieval_thoughts,
    select_prompt_documents,
)


class BaseOutputParser(ABC):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def parse(self, generated_text: str, path: List[BaseState]) -> BaseState:
        raise NotImplementedError

    def __call__(self, generated_text: str, path: List[BaseState]) -> BaseState:
        return self.parse(generated_text, path)


class AnswerGenerateOutputParser(BaseOutputParser):
    def parse(self, generated_text: str, path: List[BaseState]) -> AnswerState:
        thought_count = sum(isinstance(state, ThoughtState) for state in path)
        answer = "unknown"
        if thought_count >= self.cfg.min_num_thought:
            try:
                result = json.loads(generated_text.strip())
                answer = str(result.get("answer", "unknown"))
            except (json.JSONDecodeError, AttributeError, TypeError):
                answer = "unknown"
        return AnswerState(parent_state_id=path[-1].state_id, answer=answer)


class ThoughtGenerateOutputParser(BaseOutputParser):
    def parse(self, generated_text: str, path: List[BaseState]) -> ThoughtState:
        try:
            result = json.loads(generated_text.strip())
            thought = str(result.get("thought") or "").strip()
            query = result.get("query")
        except (json.JSONDecodeError, AttributeError, TypeError):
            thought = clean_json_value(generated_text, "thought")
            query = None
        return ThoughtState(
            parent_state_id=path[-1].state_id,
            thought=thought,
            query=str(query).strip() if query else None,
        )


class BasePromptGenerator(ABC):
    def __init__(self, cfg, template_path) -> None:
        self.cfg = cfg
        # The reference prompt files do not include a trailing newline. Keep
        # prompt tokenization identical even if a public text editor adds one.
        self.template = template_path.read_text(encoding="utf-8").rstrip("\n")

    def _values(self, path: List[BaseState]):
        _, question, thoughts, document_groups = parse_path(path)
        documents = select_prompt_documents(
            document_groups,
            source=self.cfg.prompt_document_from,
            retrieval_count=self.cfg.retrieval_count,
            maximum_count=self.cfg.prompt_max_para_count,
        )
        return {
            "question": question,
            "hints": THOUGHT_THOUGHT_DELIM.join(thoughts) if thoughts else "None",
            "documents": DOC_DOC_DELIM.join(map(str, documents)) if documents else "None",
        }

    def __call__(self, path: List[BaseState]) -> str:
        return self.template.format(**self._values(path))


class AnswerGeneratePromptGenerator(BasePromptGenerator):
    def __init__(self, cfg) -> None:
        super().__init__(cfg, cfg.answer_gen_prompt_file_path)


class ThoughtGeneratePromptGenerator(BasePromptGenerator):
    def __init__(self, cfg) -> None:
        super().__init__(cfg, cfg.thought_gen_prompt_file_path)

    def _values(self, path: List[BaseState]):
        values = super()._values(path)
        thoughts = retrieval_thoughts(path)
        values["hints"] = THOUGHT_THOUGHT_DELIM.join(thoughts) if thoughts else "None"
        return values


class GenerationStep:
    def __init__(self, cfg, generator, prompt_generator, output_parser) -> None:
        self.generator = generator
        self.prompt_generator = prompt_generator
        self.output_parser = output_parser

    def __call__(self, paths: List[List[BaseState]]) -> List[BaseState]:
        prompts = [self.prompt_generator(path) for path in paths]
        generated_texts = self.generator(prompts)
        return [
            self.output_parser(text, path)
            for text, path in zip(generated_texts, paths)
        ]
