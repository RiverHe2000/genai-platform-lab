"""Tools the agent may call. Each tool declares a Pydantic argument model (validated before
execution and rendered as JSON schema in the system prompt) and a risk level; ``high``
risk tools go through the human-approval interrupt."""
