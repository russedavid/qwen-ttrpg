"""System-neutral supervision contracts and source checks shared by data and evals."""

from __future__ import annotations

import copy
import re
from typing import Literal

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(Record):
    turn: int = Field(ge=1)
    quote: str = Field(min_length=1)


class Event(Record):
    kind: Literal[
        "entity", "fact", "claim", "resource", "action", "resolve", "knowledge"
    ]
    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str | int | float | bool | None
    delta: int | None = None
    stage: Literal["hypothetical", "declared", "requested", "reported", "established"]
    visibility: Literal["public", "private"] = "public"
    evidence: list[Evidence] = Field(min_length=1, max_length=12)
    resolves: str | None = None
    supersedes: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.kind == "resource":
            if self.stage != "established":
                raise ValueError("Resource changes require an established outcome.")
            if (self.delta is None) == (self.value is None):
                raise ValueError("Use a resource total or a delta, never both.")
            if self.value is not None and type(self.value) is not int:
                raise ValueError("Resource totals must be integers.")
        elif self.delta is not None or self.value is None:
            raise ValueError(
                "Non-resource events need a value and cannot have a delta."
            )
        if self.kind == "action" and self.stage == "established":
            raise ValueError("Describe a completed outcome as a fact or resolution.")
        if (
            self.kind in {"entity", "fact", "resolve", "knowledge"}
            and self.stage != "established"
        ):
            raise ValueError("State changes require an established outcome.")
        if self.kind == "claim" and self.stage != "reported":
            raise ValueError("A reported claim is not an established fact.")
        if (self.kind == "resolve") != bool(self.resolves):
            raise ValueError("Only resolution events identify an action they resolve.")
        return self


class Extraction(Record):
    events: list[Event] = Field(default_factory=list, max_length=50)
    uncertainties: list[str] = Field(default_factory=list)


class Citation(Record):
    id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ToolCall(Record):
    tool: str = Field(min_length=1)
    arguments: dict


class RulesAnswer(Record):
    answer: str = Field(min_length=1)
    citations: list[Citation]
    calculation: ToolCall | None = None
    missing_information: list[str]


class Narration(Record):
    narration: str = Field(min_length=1)


class PlayerReply(Record):
    utterance: str = Field(min_length=1, max_length=3000)
    recipient: str = Field(default="table", min_length=1)


MODELS = {
    "classifier": Extraction,
    "rules": RulesAnswer,
    "storyteller": Narration,
    "player": PlayerReply,
}


def exact_quote(source, quote):
    if not quote.strip():
        raise ValueError("Empty source quote.")
    match = re.search(r"\s+".join(re.escape(word) for word in quote.split()), source)
    if not match:
        raise ValueError("A quotation is not present in the supplied source.")
    return match.group(0)


def output_schema(task):
    """Match sorted training serialization; nullable fields are still explicit."""
    schema = copy.deepcopy(MODELS[task].model_json_schema())

    def visit(item):
        if isinstance(item, dict):
            if "properties" in item:
                item["properties"] = dict(sorted(item["properties"].items()))
                item["required"] = list(item["properties"])
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(schema)
    return schema


def validate_target(task, body, target):
    parsed = MODELS[task].model_validate(target)
    if task == "classifier":
        target_turns = {t["turn"] for t in body["target_turns"]}
        lookup = {
            t["turn"]: t["text"]
            for t in body.get("context_only", []) + body["target_turns"]
        }
        prior = {e["id"]: e for e in body.get("prior_events", [])}
        for event in parsed.events:
            if not target_turns.intersection(e.turn for e in event.evidence):
                raise ValueError(
                    "Each event must cite the target window, not only earlier context."
                )
            for evidence in event.evidence:
                if evidence.turn not in lookup:
                    raise ValueError(
                        "An event cites a turn outside its supplied window."
                    )
                exact_quote(lookup[evidence.turn], evidence.quote)
            for reference in [event.resolves, event.supersedes]:
                if reference and reference not in prior:
                    raise ValueError("Event links must refer to supplied prior events.")
            if event.resolves and prior[event.resolves].get("kind") != "action":
                raise ValueError("A resolution must link to a pending action.")
    elif task == "rules":
        lookup = {r["id"]: r["text"] for r in body["rules"]}
        for citation in parsed.citations:
            if citation.id not in lookup:
                raise ValueError("A citation points to an unsupplied rule excerpt.")
            exact_quote(lookup[citation.id], citation.quote)
        if not parsed.citations and not parsed.missing_information:
            raise ValueError(
                "An answer needs cited evidence or an explicit information gap."
            )
        if parsed.calculation:
            if parsed.missing_information:
                raise ValueError("Missing inputs must be resolved before a tool call.")
            tools = {t["name"]: t for t in body.get("tools", [])}
            call = parsed.calculation
            if call.tool not in tools:
                raise ValueError("The calculation tool was not declared in this input.")
            try:
                Draft202012Validator(tools[call.tool]["parameters"]).validate(
                    call.arguments
                )
            except ValidationError as exc:
                raise ValueError(
                    "Calculation arguments violate the declared tool schema."
                ) from exc
            # Tool inputs are supplied by the caller, never inferred from prose.
            if call.arguments != body.get("tool_inputs", {}).get(call.tool):
                raise ValueError(
                    "Calculation arguments must equal the explicit supplied inputs."
                )
    elif task == "player":
        recipients = {
            r["id"] for r in body.get("context", {}).get("available_recipients", [])
        } or {"table", "facilitator"}
        if parsed.recipient not in recipients:
            raise ValueError(
                "The player selected a recipient outside the supplied perspective."
            )
    return parsed.model_dump()
