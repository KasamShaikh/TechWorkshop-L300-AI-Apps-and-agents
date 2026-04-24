import logging
import os
from collections.abc import AsyncIterable
from typing import Any, Literal, Annotated
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from agent_framework import (
    AgentSession,
    ChatContext,
    Agent,
    tool,
)
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

logger = logging.getLogger(__name__)
load_dotenv()


# region Chat Service Configuration


def get_chat_client() -> OpenAIChatClient:
    """Return Azure OpenAI chat client using the v1 API with managed identity."""
    endpoint = os.getenv("gpt_endpoint")
    deployment_name = os.getenv("gpt_deployment")

    if not endpoint:
        raise ValueError("gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")

    return OpenAIChatClient(
        model=deployment_name,
        base_url=f"{endpoint.rstrip('/')}/openai/v1/",
        credential=DefaultAzureCredential(),
    )


# endregion


# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


# endregion


# region Tools


@tool(
    name="get_products",
    description="Retrieves a set of products based on a natural language user query.",
)
def get_products(
    query: Annotated[str, "A natural language query to search for products."],
) -> dict[str, Any]:
    """Retrieve a set of products based on the user's natural language query."""
    logger.info(f"Retrieving products for query: {query}")
    # Hardcoded sample product catalog for demo purposes
    return {
        "products": [
            {
                "id": "P001",
                "name": "Eco-Friendly Paint Roller",
                "description": "A high-quality paint roller made from recycled materials, perfect for smooth and even paint application.",
                "price": 15.99,
            },
            {
                "id": "P002",
                "name": "Premium Paint Brush Set",
                "description": "A set of 5 premium paint brushes in various sizes, ideal for both professional and DIY painting projects.",
                "price": 25.49,
            },
            {
                "id": "P003",
                "name": "All-Purpose Paint Tray",
                "description": "A durable and versatile paint tray suitable for use with rollers and brushes, featuring a non-slip base.",
                "price": 9.99,
            },
        ]
    }


# endregion


# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework-based agents to handle Zava product management tasks."""

    agent: Agent
    session: AgentSession = None
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        # Configure the chat completion service explicitly
        chat_service = get_chat_client()

        # Specialist agent: generates marketing copy for given products
        marketing_agent = Agent(
            client=chat_service,
            name="MarketingAgent",
            instructions=(
                "You are a marketing specialist for Zava. Given a set of products, "
                "craft compelling, friendly, and persuasive marketing descriptions that "
                "highlight unique benefits and encourage customers to buy. Return only "
                "the updated marketing copy for each product."
            ),
        )

        # Specialist agent: ranks products according to customer intent
        ranker_agent = Agent(
            client=chat_service,
            name="RankerAgent",
            instructions=(
                "You are a product ranking specialist. Given a set of products and a user "
                "request, rank the products from most to least relevant to the user's needs. "
                "Briefly justify why the top choice is the best fit."
            ),
        )

        # Specialist agent: retrieves products using the get_products tool
        product_agent = Agent(
            client=chat_service,
            name="ProductAgent",
            instructions=(
                "You are a product catalog specialist for Zava. When the user asks about "
                "products, you MUST use the get_products tool to retrieve the current product "
                "catalog. Summarize the products clearly and return them to the caller."
            ),
            tools=get_products,
        )

        # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        self.agent = Agent(
            client=chat_service,
            name="ProductManagerAgent",
            instructions=(
                "Your role is to carefully analyze the user's request and delegate to the appropriate specialist agent(s):\n"
                "- Use the ProductAgent tool to look up Zava products whenever the user asks about the catalog, pricing, or availability.\n"
                "- Use the MarketingAgent tool when the user wants marketing copy, descriptions, or persuasive content about products.\n"
                "- Use the RankerAgent tool when the user wants a recommendation or ranking of the best product for their needs.\n"
                "You may call multiple specialist tools in sequence to compose the final answer.\n\n"
                "IMPORTANT: You must ALWAYS respond with a valid JSON object in the following format:\n"
                '{"status": "<status>", "message": "<your response>"}\n\n'
                'Where status is one of: "input_required", "completed", or "error".\n'
                '- Use "input_required" when you need more information from the user.\n'
                '- Use "completed" when the task is finished.\n'
                '- Use "error" when something went wrong.\n\n'
                "Never respond with plain text. Always use the JSON format above."
            ),
            tools=[
                product_agent.as_tool(),
                marketing_agent.as_tool(),
                ranker_agent.as_tool(),
            ],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        """Handle synchronous tasks (like tasks/send)."""
        await self._ensure_session_exists(session_id)

        # Use Agent Framework's run for a single shot
        response = await self.agent.run(
            messages=user_input,
            session=self.session,
            options=OpenAIChatOptions(response_format=ResponseFormat),
        )
        return self._get_agent_response(response.text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """For streaming tasks we yield the Agent Framework agent's run_stream progress."""
        await self._ensure_session_exists(session_id)

        chunks: list[ChatContext] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            session=self.session,
        ):
            if chunk.text:
                chunks.append(chunk.text)

        if chunks:
            yield self._get_agent_response(sum(chunks[1:], chunks[0]))

    def _get_agent_response(self, message: ChatContext) -> dict[str, Any]:
        """Extracts the structured response from the agent's message content."""
        structured_response = None
        try:
            structured_response = ResponseFormat.model_validate_json(message)
        except ValidationError:
            logger.info("Message did not come in JSON format.")
            default_response = {
                "is_task_complete": True,
                "require_user_input": False,
                "content": message,
            }
        except Exception:
            logger.error("An unexpected error occurred while processing the message.")
            default_response = {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "We are unable to process your request at the moment. Please try again.",
            }

        if structured_response and isinstance(structured_response, ResponseFormat):
            response_map = {
                "input_required": {
                    "is_task_complete": False,
                    "require_user_input": True,
                },
                "error": {
                    "is_task_complete": False,
                    "require_user_input": True,
                },
                "completed": {
                    "is_task_complete": True,
                    "require_user_input": False,
                },
            }

            response = response_map.get(structured_response.status)
            if response:
                return {**response, "content": structured_response.message}

        return default_response

    async def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure the session exists for the given session ID."""
        if self.session is None or self.session.service_session_id != session_id:
            self.session = self.agent.create_session(session_id=session_id)


# endregion
