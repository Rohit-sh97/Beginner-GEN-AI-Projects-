from langchain_core.prompts import ChatPromptTemplate


research_prompt = ChatPromptTemplate.from_template("""
You are an AI research assistant.

Prepare a clear introductory research overview about this topic:
{topic}

Return:
- topic: the research topic
- overview: a concise explanation
- key_points: 5 important points
- conclusion: a short conclusion

Be accurate. Do not invent statistics, citations, or sources.
If the topic involves recent events, explain that current sources
should be checked before treating the information as up to date.
""")