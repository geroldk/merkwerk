"""Typed API transport for cross-project moves on a local Git host (I1).

Only a Git host serves these routes; elsewhere they answer 404 (kind "not_git_host").
A refusal carries `detail.kind`; nothing was moved then.
"""

from httpx import AsyncClient

from basic_memory.schemas.git_moves import (
    GitCrossProjectMoveRequest,
    GitCrossProjectMoveResult,
    GitMoveCatchUp,
)


class GitMoveClient:
    def __init__(self, client: AsyncClient, project_id: str) -> None:
        """`project_id` is the source project's external UUID."""
        self.client = client
        self.path = f"/v2/projects/{project_id}/git/moves"

    async def move(self, request: GitCrossProjectMoveRequest) -> GitCrossProjectMoveResult:
        from basic_memory.mcp.tools.utils import call_post

        response = await call_post(
            self.client,
            self.path,
            json=request.model_dump(mode="json"),
            client_name="git_moves",
            operation="move",
            path_template="/v2/projects/{project_id}/git/moves",
        )
        return GitCrossProjectMoveResult.model_validate(response.json())

    async def catch_up(self) -> GitMoveCatchUp:
        """Bring this project's SQL to its head (the client's project is the target here)."""
        from basic_memory.mcp.tools.utils import call_post

        response = await call_post(
            self.client,
            f"{self.path}/catch-up",
            client_name="git_moves",
            operation="catch_up",
            path_template="/v2/projects/{project_id}/git/moves/catch-up",
        )
        return GitMoveCatchUp.model_validate(response.json())
