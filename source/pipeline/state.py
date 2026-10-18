"""State objects used by the iterative QA controller."""

from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4

from source.module.index.docstore import Document


@dataclass
class BaseState:
    parent_state_id: Optional[str] = None
    state_id: str = field(default_factory=lambda: str(uuid4()), init=False)

    def to_logging_format(self) -> str:
        return type(self).__name__


@dataclass
class QuestionState(BaseState):
    question_id: Optional[str] = None
    question: Optional[str] = None
    ground_truth: Optional[str] = None

    def to_logging_format(self) -> str:
        return f"({self.question_id}) Q: {self.question}"


@dataclass
class ThoughtState(BaseState):
    thought: Optional[str] = None
    query: Optional[str] = None

    def to_logging_format(self) -> str:
        return f"Thought: {self.thought}" + (f"\nNext: {self.query}" if self.query else "")


@dataclass
class AnswerState(BaseState):
    answer: Optional[str] = None

    def to_logging_format(self) -> str:
        return f"Answer: {self.answer}"


@dataclass
class DocumentState(BaseState):
    documents: Optional[List[Document]] = None

    def to_logging_format(self) -> str:
        titles = [document.metadata.get("title", "") for document in self.documents or []]
        return f"Documents: {titles}"


@dataclass
class ResumeState(BaseState):
    pass


@dataclass
class EndState(BaseState):
    answer: Optional[str] = None
    question_id: Optional[str] = None

    def to_logging_format(self) -> str:
        return f"End: {self.answer}"
