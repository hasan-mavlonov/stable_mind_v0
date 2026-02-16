from stablemind.agent import StableMindAgent

agent = StableMindAgent(root_dir=".")
agent.step("France cafe was loud and chaotic today")
agent.step("France cafe was loud again")
agent.step("France cafe is never quiet")