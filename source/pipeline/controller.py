"""Controller for the iterative retrieval-generation state machine."""

import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from source.pipeline.state import BaseState, EndState, QuestionState


class PipelineController:
    def __init__(
        self,
        pipeline: List,
        logging_file_path: Path,
        prediction_file_path: Path,
        retain_history: bool = True,
    ):
        self.pipeline = pipeline
        self.logging_file_path = Path(logging_file_path)
        self.prediction_file_path = Path(prediction_file_path)
        self.state_tree: Dict[str, BaseState] = {}
        self.running_state_ids: List[str] = []
        self.end_state_ids: List[str] = []
        self.logs: Dict[str, List[str]] = {}
        self.retain_history = retain_history

    def run(self, start_states: List[QuestionState], batch_size: int = 1):
        predictions = {}
        for start in tqdm(range(0, len(start_states), batch_size), desc="Questions"):
            self._update(start_states[start : start + batch_size])
            while self.running_state_ids:
                for step in self.pipeline:
                    self._update(step(self._next_paths()))
            predictions.update(
                {
                    self.state_tree[state_id].question_id: str(
                        self.state_tree[state_id].answer
                    )
                    for state_id in self.end_state_ids
                }
            )
            if not self.retain_history:
                # Release state objects, but keep text logs for the final output.
                self.state_tree.clear()
                self.running_state_ids.clear()
                self.end_state_ids.clear()
        return self._save_results(predictions)

    def _path(self, state_id: str) -> List[BaseState]:
        path = []
        while state_id is not None:
            state = self.state_tree[state_id]
            path.append(state)
            state_id = state.parent_state_id
        return list(reversed(path))

    def get_path(self, state_id: str) -> List[BaseState]:
        """Return the root-to-state path for evaluation and inspection."""

        return self._path(state_id)

    def _next_paths(self) -> List[List[BaseState]]:
        paths = [self._path(state_id) for state_id in self.running_state_ids]
        self.running_state_ids.clear()
        return paths

    def _update(self, states: List[BaseState]) -> None:
        if self.running_state_ids:
            raise RuntimeError("Pipeline states must be consumed before the next update.")
        for state in states:
            if state.state_id in self.state_tree:
                if not isinstance(state, EndState):
                    self.running_state_ids.append(state.state_id)
                continue
            self.state_tree[state.state_id] = state
            path = self._path(state.state_id)
            question = next(item for item in path if isinstance(item, QuestionState))
            self.logs.setdefault(question.question_id, []).append(state.to_logging_format())
            if isinstance(state, EndState):
                state.question_id = question.question_id
                self.end_state_ids.append(state.state_id)
            else:
                self.running_state_ids.append(state.state_id)

    def _save_results(self, predictions=None):
        if predictions is None:
            predictions = {
                self.state_tree[state_id].question_id: str(
                    self.state_tree[state_id].answer
                )
                for state_id in self.end_state_ids
            }
        self.logging_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.logging_file_path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.prediction_file_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return predictions
