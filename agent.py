"""
Exercise-generation agent (qwen3:4b via Ollama, orchestrated with LangGraph).
"""

import uuid
from typing import Callable, Literal, Optional, TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# Replies that end the session
STOP_PHRASES = {
    "no", "nope", "no thanks", "no thank you", "nothing", "nothing else", "no more",
    "done", "i'm done", "im done", "that's all", "thats all", "that's it", "thats it",
    "finish", "finished", "stop", "quit", "exit", "bye", "goodbye", "all good", "i'm good",
}

ExerciseType = Literal["multiple_choice", "gap_fill", "true_false", "short_answer"]


class ExerciseSpec(BaseModel):
    """What the teacher wants, parsed from their free-text reply."""
    types: list[ExerciseType] = Field(default_factory=lambda: ["multiple_choice", "gap_fill"])
    per_type: int = Field(5, description="Number of exercises of each type")
    focus: str = Field("", description="Topic or skill focus, e.g. 'phrasal verbs', 'vocabulary'")
    level: Optional[str] = Field(None, description="CEFR level A1-C2 if the teacher asked for a specific one")
    finished: bool = Field(
        False,
        description="True ONLY if the teacher says they want nothing else and are done. False if they ask for any exercises.",
    )


class Exercise(BaseModel):
    type: ExerciseType
    question: str
    options: list[str] = Field(default_factory=list, description="Only for multiple_choice")
    answer: str


class ExerciseSet(BaseModel):
    exercises: list[Exercise]


class State(TypedDict, total=False):
    # inputs
    clean_text: str
    vocab: dict
    level: str
    # current round
    teacher_reply: str
    spec: dict
    exercises: list
    issues: list
    attempts: int
    final: str            # markdown of the latest set (shown to the teacher)
    # accumulated over all rounds
    batches: list         # markdown of every set delivered so far
    all_exercises: list   # every exercise delivered so far
    past_questions: list  # used to avoid repeating questions in later sets


# Interaction
def console_ask(question: str) -> str:
    return input("\n" + question + "\n> ")


def console_show(text: str) -> None:
    print("\n" + text)


# Agent
class ExerciseAgent:
    def __init__(
        self,
        model: str = "qwen3:4b",
        ask_fn: Callable[[str], str] = console_ask,
        display_fn: Callable[[str], None] = console_show,
        max_attempts: int = 2,
        max_prompt_chars: int = 12_000,
        max_vocab_per_pos: int = 20,
        num_ctx: int = 8192,
    ):
        """
        ask_fn(question) -> teacher's reply (str). Called whenever the agent needs input.
        display_fn(markdown) -> None. Called to show each finished set of exercises.
        max_attempts: generation attempts per set before delivering anyway.
        max_prompt_chars / max_vocab_per_pos: caps on what is sent to the small model.
        num_ctx: Ollama's default context is small and silently truncates prompts, so set it explicitly.
        """
        self.ask_fn = ask_fn
        self.display_fn = display_fn
        self.max_attempts = max_attempts
        self.max_prompt_chars = max_prompt_chars
        self.max_vocab_per_pos = max_vocab_per_pos
        self.llm = ChatOllama(model=model, temperature=0, num_ctx=num_ctx, reasoning=False)
        self.gen_llm = ChatOllama(model=model, temperature=0.3, num_ctx=num_ctx, reasoning=False)
        self.graph = self._build_graph()

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    @staticmethod
    def _step(result: dict) -> dict:
        """Turn a graph result into {'show', 'question', 'done'}."""
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            return {"show": payload.get("show"), "question": payload["question"], "done": False}
        return {"show": None, "question": None, "done": True}

    #Begin a session
    def start(self, transcript: str, vocab: dict, level: str) -> tuple[str, dict]:
        thread_id = str(uuid.uuid4())
        result = self.graph.invoke(
            {"clean_text": transcript, "vocab": self._trim_vocab(vocab), "level": level, "attempts": 0},
            self._config(thread_id),
        )
        return thread_id, self._step(result)

    def send(self, thread_id: str, reply: str) -> dict:
        result = self.graph.invoke(Command(resume=reply), self._config(thread_id))
        return self._step(result)

    def get_result(self, thread_id: str) -> dict:
        values = self.graph.get_state(self._config(thread_id)).values
        batches = values.get("batches", [])
        return {
            "markdown": "\n\n---\n\n".join(batches),
            "batches": batches,
            "exercises": values.get("all_exercises", []),
        }

    def run(self, transcript: str, vocab: dict, level: str) -> dict:
        thread_id, step = self.start(transcript, vocab, level)
        while not step["done"]:
            if step["show"]:
                self.display_fn(step["show"])
            step = self.send(thread_id, self.ask_fn(step["question"]))
        return self.get_result(thread_id)


    def _trim_vocab(self, vocab: dict) -> dict:
        """Keep the prompt small: at most N words per part of speech."""
        n = self.max_vocab_per_pos
        trimmed = {}
        for pos, words in vocab.items():
            if isinstance(words, dict):  # word to count
                words = [w for w, _ in sorted(words.items(), key=lambda kv: -kv[1])]
            trimmed[pos] = list(words)[:n]
        return trimmed

    @staticmethod
    def _is_stop(reply: str) -> bool:
        text = reply.strip().lower().replace("\u2019", "'").strip(" .!?,")
        return text in STOP_PHRASES

    # Human-in-the-loop + LLM: parse the teacher's reply
    def ask_teacher(self, state: State):
        first_round = not state.get("batches")
        question = (
            "What kind of exercises would you like? (types, how many, focus, level)"
            if first_round
            else "Would you like any other exercises? Describe them, or say 'done' to finish."
        )

        reply = interrupt({"question": question, "show": state.get("final")})

        if self._is_stop(reply):
            spec = ExerciseSpec(finished=True)
        else:
            parser = self.llm.with_structured_output(ExerciseSpec, method="json_schema")
            try:
                spec = parser.invoke(
                    "Convert the teacher's request into an exercise specification. "
                    "Allowed types: multiple_choice, gap_fill, true_false, short_answer. "
                    "Map anything else (e.g. comprehension questions -> short_answer, "
                    "vocabulary -> gap_fill or multiple_choice with focus='vocabulary'). "
                    "Leave fields at their defaults if the teacher did not mention them. "
                    "Set finished=true ONLY if the teacher says they want nothing else "
                    "(e.g. 'no thanks', \"that's all\", 'I am done'); if they ask for any exercises, finished=false.\n\n"
                    f"Teacher's request: {reply!r}"
                )
            except Exception:
                spec = ExerciseSpec()  # fall back to defaults if parsing fails
        # New request: reset the retry counter and problems from the previous round.
        return {"teacher_reply": reply, "spec": spec.model_dump(), "attempts": 0, "issues": []}

    def route_after_ask(self, state: State) -> str:
        return "finish" if state["spec"]["finished"] else "generate_exercises"

    # Generate LLM
    TYPE_RULES = {
        "multiple_choice": '3-4 options; "answer" must be exactly one of the options.',
        "gap_fill": 'the question is a sentence from the transcript with the target word replaced by ____; "answer" is that word.',
        "true_false": '"answer" is exactly "True" or "False". Mix true and false statements.',
        "short_answer": 'a comprehension question; "answer" is a brief model answer.',
    }

    def generate_exercises(self, state: State):
        spec = state["spec"]
        level = spec.get("level") or state["level"]
        past = list(state.get("past_questions", []))
        structured = self.gen_llm.with_structured_output(ExerciseSet, method="json_schema")
        all_new = []

        for t in spec["types"]:
            prompt = f"""Create language-learning exercises based ONLY on the transcript below.
    CEFR level of the learner: {level}. Use vocabulary and sentence complexity suited to that level.

    Create exactly {spec['per_type']} exercises of type "{t}" and nothing else.
    Rule for {t}: {self.TYPE_RULES[t]}
    Teacher's focus: {spec['focus'] or 'none'}.
    Candidate vocabulary from the transcript: {state['vocab']}
    """
            if past or all_new:
                prev = past[-40:] + [e["question"] for e in all_new]
                prompt += "\nDo NOT repeat or closely paraphrase these questions:\n- " + "\n- ".join(prev) + "\n"
            relevant = [i for i in state.get("issues", []) if f"({t})" in i or t in i]
            if relevant:
                prompt += "\nYour previous attempt had these problems. Fix them:\n- " + "\n- ".join(relevant) + "\n"
            prompt += f"\nTranscript:\n{state['clean_text'][:self.max_prompt_chars]}"

            result = structured.invoke(prompt)
            batch = [e.model_dump() for e in result.exercises][:spec["per_type"]]
            for e in batch:
                e["type"] = t  # force the type, don't trust the model
            all_new += batch

        return {"exercises": all_new, "attempts": state.get("attempts", 0) + 1}

    # Python validation
    def validate(self, state: State):
        spec, exercises = state["spec"], state["exercises"]
        text = state["clean_text"].lower()
        issues = []

        for t in spec["types"]:
            n = sum(1 for e in exercises if e["type"] == t)
            if n != spec["per_type"]:
                issues.append(f"Expected {spec['per_type']} {t} exercises but got {n}.")

        seen = set()
        for i, e in enumerate(exercises, 1):
            label = f"Exercise {i} ({e['type']})"
            q, a = e["question"].strip(), e["answer"].strip()
            if not q or not a:
                issues.append(f"{label}: empty question or answer.")
            if q.lower() in seen:
                issues.append(f"{label}: duplicate question.")
            seen.add(q.lower())

            if e["type"] == "multiple_choice":
                if not 3 <= len(e["options"]) <= 4:
                    issues.append(f"{label}: needs 3-4 options.")
                if a not in e["options"]:
                    issues.append(f"{label}: answer is not one of the options.")
            elif e["type"] == "gap_fill":
                if "____" not in q:
                    issues.append(f"{label}: question must contain ____.")
                if a.lower() not in text:
                    issues.append(f"{label}: answer '{a}' does not appear in the transcript.")
            elif e["type"] == "true_false" and a not in ("True", "False"):
                issues.append(f"{label}: answer must be exactly 'True' or 'False'.")

        return {"issues": issues}

    def route_after_validate(self, state: State) -> str:
        if state["issues"] and state["attempts"] < self.max_attempts:
            return "generate_exercises"
        return "deliver"

    # Delivery
    def deliver(self, state: State):
        spec = state["spec"]
        level = spec.get("level") or state["level"]
        batches = state.get("batches", [])
        lines = [f"# Exercise set {len(batches) + 1} (CEFR {level})", ""]

        n = 0
        for t in spec["types"]:
            group = [e for e in state["exercises"] if e["type"] == t]
            if not group:
                continue
            lines += [f"## {t.replace('_', ' ').title()}", ""]
            for e in group:
                n += 1
                lines.append(f"{n}. {e['question']}")
                for letter, opt in zip("abcd", e["options"]):
                    lines.append(f"   {letter}) {opt}")
                lines.append("")

        lines += ["## Answer key", ""]
        n = 0
        for t in spec["types"]:
            for e in (x for x in state["exercises"] if x["type"] == t):
                n += 1
                lines.append(f"{n}. {e['answer']}")

        if state["issues"]:
            lines += ["", "> Note: automatic checks still found these problems, please review:"]
            lines += [f"> - {i}" for i in state["issues"]]

        markdown = "\n".join(lines)
        return {
            "final": markdown,
            "batches": batches + [markdown],
            "all_exercises": state.get("all_exercises", []) + state["exercises"],
            "past_questions": state.get("past_questions", []) + [e["question"] for e in state["exercises"]],
        }

    # Graph
    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("ask_teacher", self.ask_teacher)
        g.add_node("generate_exercises", self.generate_exercises)
        g.add_node("validate", self.validate)
        g.add_node("deliver", self.deliver)

        g.add_edge(START, "ask_teacher")
        g.add_conditional_edges("ask_teacher", self.route_after_ask,
                                {"generate_exercises": "generate_exercises", "finish": END})
        g.add_edge("generate_exercises", "validate")
        g.add_conditional_edges("validate", self.route_after_validate,
                                {"generate_exercises": "generate_exercises", "deliver": "deliver"})
        g.add_edge("deliver", "ask_teacher")  # loop: ask again after every set

        return g.compile(checkpointer=MemorySaver())  # a checkpointer is required for interrupt()