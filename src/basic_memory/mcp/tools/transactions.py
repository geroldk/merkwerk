"""Explicit transaction tool; ordinary bm_* writes remain implicit single commits."""

import json
from typing import Annotated, Literal
from uuid import UUID

from fastmcp import Context
from pydantic import Field

from basic_memory.mcp.clients.git_transactions import GitTransactionClient
from basic_memory.mcp.project_context import get_project_client
from basic_memory.mcp.refusals import api_refusals, refusal
from basic_memory.mcp.server import mcp
from basic_memory.mcp.tool_params import (
    OutputFormat,
    parse_project,
    routing_arguments,
)
from basic_memory.schemas.git_transactions import (
    GitTransactionCommit,
    GitTransactionControl,
    GitTransactionDelete,
    GitTransactionMove,
    GitTransactionStart,
    GitTransactionWrite,
)

type TransactionCommand = Literal[
    "start", "status", "list", "read", "write", "move", "delete", "keep_alive", "commit", "rollback"
]


@mcp.tool(
    name="bm_transaction",
    title="Git transaction",
    description=(
        "Local Git host only. Start a private fixed-base transaction; keep its ID/key. "
        "Use list/read (view=base or draft), write complete canonical Markdown (or append/prepend), "
        "move/delete, keep_alive, status, commit or rollback. Every call requires project. "
        "Changes/commit/rollback require expected_revision. No automatic retry of lost writes. "
        "For reviewed conflict transfer, start with source_transaction_id, source_key and source_revision; "
        "the source is frozen/protected and the new draft starts on current Git without copying old text."
    ),
    tags={"git", "transactions"},
    annotations={
        "title": "Git transaction",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def bm_transaction(
    command: Annotated[TransactionCommand, Field(description="Transaction operation.")],
    transaction_id: Annotated[UUID | None, Field(description="ID returned by start.")] = None,
    transaction_key: Annotated[
        str | None, Field(description="Private key returned only by start.")
    ] = None,
    expected_revision: Annotated[
        int | None,
        Field(ge=0, description="Last read draft revision; required for changes/commit/rollback."),
    ] = None,
    identifier: Annotated[
        str | None,
        Field(description="Read: exact path, UUID or permalink. Changes: exact Markdown path."),
    ] = None,
    content: Annotated[
        str | None,
        Field(description="For write: full canonical Markdown, or appended/prepended text."),
    ] = None,
    destination: Annotated[
        str | None,
        Field(description="For move: new exact Markdown path; UUID/permalink are preserved."),
    ] = None,
    operation: Annotated[
        Literal["replace", "append", "prepend"], Field(description="Write behavior.")
    ] = "replace",
    view: Annotated[
        Literal["base", "draft"], Field(description="For list/read: fixed basis or own draft.")
    ] = "draft",
    message: Annotated[
        str,
        Field(
            description=(
                "For commit: Git message. It must not contain 'Basic-Memory-' in any case: "
                "BM reserves that prefix for its own commit trailers."
            )
        ),
    ] = "Publish Basic Memory transaction",
    source_transaction_id: Annotated[
        UUID | None,
        Field(
            description="For start: freeze/protect this old source; do not copy it automatically."
        ),
    ] = None,
    source_key: Annotated[
        str | None, Field(description="For start with a source: its private key.")
    ] = None,
    source_revision: Annotated[
        int | None, Field(ge=0, description="For start with a source: its reviewed revision.")
    ] = None,
    output_format: OutputFormat = "json",
    project: Annotated[
        str | None, Field(description="Required local Git project name or UUID.")
    ] = None,
    context: Context | None = None,
) -> str | dict[str, object]:
    if project is None:
        raise refusal(
            "INVALID_ARGUMENTS", "Transaction calls need project.", output_format=output_format
        )
    if command != "start" and (transaction_id is None or transaction_key is None):
        raise refusal(
            "INVALID_ARGUMENTS",
            "This command needs transaction_id and transaction_key.",
            output_format=output_format,
        )
    if command in {"write", "move", "delete", "commit", "rollback"} and expected_revision is None:
        raise refusal(
            "INVALID_ARGUMENTS",
            "Inspect the draft and supply expected_revision.",
            output_format=output_format,
        )
    if command in {"read", "write", "move", "delete"} and identifier is None:
        raise refusal(
            "INVALID_ARGUMENTS", "This command needs identifier.", output_format=output_format
        )
    if command == "write" and content is None or command == "move" and destination is None:
        raise refusal(
            "INVALID_ARGUMENTS",
            "write needs content; move needs destination.",
            output_format=output_format,
        )
    project_name, project_id = routing_arguments(parse_project(project))
    with api_refusals(output_format):
        async with get_project_client(project_name, context=context, project_id=project_id) as (
            client,
            active_project,
        ):
            transactions = GitTransactionClient(client, active_project.external_id)
            if command == "start":
                result = await transactions.start(
                    GitTransactionStart(
                        source_transaction_id=source_transaction_id,
                        source_key=source_key,
                        source_revision=source_revision,
                    )
                )
            else:
                assert transaction_id is not None and transaction_key is not None
                match command:
                    case "status" | "keep_alive":
                        result = await transactions.status(
                            transaction_id, transaction_key, keep_alive=command == "keep_alive"
                        )
                    case "list":
                        result = await transactions.listing(
                            transaction_id, transaction_key, view=view
                        )
                    case "read":
                        assert identifier is not None
                        result = await transactions.read(
                            transaction_id, transaction_key, identifier, view=view
                        )
                    case "write" | "move" | "delete" | "commit" | "rollback":
                        assert expected_revision is not None
                        match command:
                            case "write":
                                assert identifier is not None and content is not None
                                request = GitTransactionWrite(
                                    expected_revision=expected_revision,
                                    path=identifier,
                                    content=content,
                                    operation=operation,
                                )
                            case "move":
                                assert identifier is not None and destination is not None
                                request = GitTransactionMove(
                                    expected_revision=expected_revision,
                                    path=identifier,
                                    destination=destination,
                                )
                            case "delete":
                                assert identifier is not None
                                request = GitTransactionDelete(
                                    expected_revision=expected_revision, path=identifier
                                )
                            case "commit":
                                request = GitTransactionCommit(
                                    expected_revision=expected_revision, message=message
                                )
                            case "rollback":
                                request = GitTransactionControl(expected_revision=expected_revision)
                        result = await transactions.change(transaction_id, transaction_key, request)
    payload = result.model_dump(mode="json")
    return payload if output_format == "json" else json.dumps(payload, ensure_ascii=False, indent=2)
