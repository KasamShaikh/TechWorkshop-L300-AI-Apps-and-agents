from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy
import httpx
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Azure AI Project Information
azure_ai_project = os.getenv("FOUNDRY_ENDPOINT")

# Instantiate your AI Red Teaming Agent with custom attack prompts
# (The default built-in objectives configuration is shown below, commented out.)
# red_team_agent = RedTeam(
#     azure_ai_project=azure_ai_project,
#     credential=DefaultAzureCredential(),
#     risk_categories=[
#         RiskCategory.Violence,
#         RiskCategory.HateUnfairness,
#         RiskCategory.Sexual,
#         RiskCategory.SelfHarm
#     ],
#     num_objectives=5,
# )
red_team_agent = RedTeam(
    azure_ai_project=azure_ai_project,
    credential=DefaultAzureCredential(),
    custom_attack_seed_prompts="data/custom_attack_prompts.json",
)

# Configuration to target the Cora agent via Foundry Responses API
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")

foundry_endpoint = os.environ.get("FOUNDRY_ENDPOINT").rstrip("/")
model = os.environ.get("gpt_deployment")


def cora_target(query: str) -> str:
    """Send a prompt to the Cora agent and return the response text."""
    token = token_provider()
    response = httpx.post(
        f"{foundry_endpoint}/openai/v1/responses",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={
            "model": model,
            "input": query,
            "agent_reference": {"name": "cora", "type": "agent_reference"},
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()

    # Extract text from the Responses API output
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return data.get("output_text", str(data))


async def main():
    red_team_result = await red_team_agent.scan(
        target=cora_target,
        scan_name="Red Team Scan - Easy-Moderate Strategies",
        attack_strategies=[
            AttackStrategy.Flip,
            AttackStrategy.ROT13,
            AttackStrategy.Base64,
            AttackStrategy.AnsiAttack,
            AttackStrategy.Tense,
        ],
    )


asyncio.run(main())
