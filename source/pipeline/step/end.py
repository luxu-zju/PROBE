"""Termination rule for iterative QA."""

from typing import List

from source.pipeline.state import AnswerState, BaseState, DocumentState, EndState, ResumeState
from source.pipeline.utils import parse_path, retrieval_thoughts


class EndStep:
    def __init__(
        self,
        cfg,
        combine_next_hop_query: bool = False,
        limit_retrieval_rounds: bool = False,
    ) -> None:
        self.cfg = cfg
        self.combine_next_hop_query = combine_next_hop_query
        self.limit_retrieval_rounds = limit_retrieval_rounds

    def __call__(self, paths: List[List[BaseState]]) -> List[BaseState]:
        next_states = []
        for path in paths:
            answers = [state.answer for state in path if isinstance(state, AnswerState)]
            known_answers = [answer for answer in answers if answer.lower() != "unknown"]
            if known_answers:
                next_states.append(
                    EndState(parent_state_id=path[-1].state_id, answer=known_answers[0])
                )
                continue

            if self.combine_next_hop_query:
                thoughts = retrieval_thoughts(path)
            else:
                _, _, thoughts, _ = parse_path(path)
            if len(thoughts) >= self.cfg.max_thought_rounds:
                next_states.append(
                    EndState(parent_state_id=path[-1].state_id, answer=" ".join(thoughts))
                )
            elif self.limit_retrieval_rounds and sum(
                isinstance(state, DocumentState) for state in path
            ) >= self.cfg.max_thought_rounds + 1:
                # T_max counts thought rounds after the initial retrieval, so
                # at most T_max + 1 retrieval/answer rounds are permitted.
                next_states.append(
                    EndState(parent_state_id=path[-1].state_id, answer=answers[-1])
                )
            else:
                next_states.append(ResumeState(parent_state_id=path[-1].state_id))
        return next_states
