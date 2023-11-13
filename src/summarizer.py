import os
import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4", api_key, org_id))

# promt = kernel.create_semantic_function("""pring hello world""")
# print(promt())

summarize = kernel.create_semantic_function("{{$input}} \n\n one line TLDR with the fewest words")
print(summarize("""The sponsor should identify risks that may have a meaningful impact on critical to 
1117 quality factors. Risks should be considered across the processes used in the clinical 
1118 trial (e.g., patient selection, informed consent process, randomisation and 
1119 investigational product administration, data handling, and service provider activities). """))