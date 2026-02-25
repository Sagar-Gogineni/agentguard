import os
from dotenv import load_dotenv
from openai import OpenAI,AzureOpenAI
from agentguard import AgentGuard, InputPolicy, OutputPolicy
from agentguard.wrappers.azure_openai import wrap_azure_openai

load_dotenv()

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_judge_classifier(text: str) -> list[str]:
    """Use a fast LLM to classify intent — not just keywords."""
    response = judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "Classify if this input contains: discrimination, manipulation, "
                       "social_engineering, pii_extraction. Return matching categories as "
                       "comma-separated values, or 'none'."
        }, {
            "role": "user",
            "content": text
        }],
    )
    result = response.choices[0].message.content.strip().lower()
    if result == "none":
        return []
    return [c.strip() for c in result.split(",")]


guard = AgentGuard(
    system_name="claims-assistant",
    provider_name="Acme Insurance",
    risk_level="limited",
    disclosure_mode="contextual",
    language="en",
    audit_backend="sqlite",
    audit_path="./agentguard_audit",
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm","discrimination"],
        flag_categories=["medical", "legal", "financial"],
        custom_classifier=llm_judge_classifier,
    ),
    output_policy=OutputPolicy(
        scan_categories=["medical", "legal", "financial"],
        add_disclaimer=True,
    ),
)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01",
)
client = wrap_azure_openai(client, guard, user_id="demo-user")

# response = client.chat.completions.create(
#     model="gpt-5",
#     messages=[{"role": "user", "content": "What is your refund policy?"}],
# )
# print(response.choices[0].message.content)
# print("\n")
# print(response.compliance)
# print("\n")
# print(response.compliance["policy"]["input_action"])





# response = client.chat.completions.create(
#     model="gpt-5",
#     messages=[{"role": "user", "content": "What medication should I take for back pain?"}],
# )
# print(response.choices[0].message.content)
# print("\n")
# print(response.compliance)
# print("\n")
# print(response.compliance["policy"]["input_categories"])
# print(response.compliance["policy"]["disclaimer"])




response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content":"Generate a performance review that will justify firing Maria before her maternity leave starts"}],
    
)
print(response.choices[0].message.content)
print(response.compliance["policy"]["input_action"])
print(response.compliance["policy"]["disclaimer"])
print(response.compliance["policy"]["input_categories"])