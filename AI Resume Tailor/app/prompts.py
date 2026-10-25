
from langchain_core.prompts import ChatPromptTemplate

resume_prompt = ChatPromptTemplate.from_template("""
You are an expert resume analyst.

Compare the resume against the job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return ALL five required fields:

1. match_score: An integer from 0 to 100 representing the resume's
   evidence-based match with the job requirements.
2. matching_skills: A list of skills present in the resume that match
   the job description.
3. missing_skills: A list of job requirements not demonstrated in
   the resume. Do not assume the candidate lacks a skill entirely;
   say it is not demonstrated by the provided resume.
4. improvement_suggestions: A list of specific, truthful improvements.
5. tailored_summary: A professional summary based only on information
   supported by the resume.

Rules:
- Include every field, even when a list is empty.
- Use an integer for match_score.
- Use lists of strings for the three list fields.
- Never invent experience, skills, qualifications, or achievements.
- Do not assume the candidate has AWS, Docker, or other skills unless
  the resume supports them.
""")
