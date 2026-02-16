from stablemind.agent import StableMindAgent

agent = StableMindAgent(root_dir=".", llm_client=None)  # uses OpenAI via default now
res = agent.step("Hey, remember that I love Turkish series.", session_id="s1")

print("TURN:", res.turn)
print("REPLY:", res.text)
print("DEBUG EVENTS:", res.debug["events"])