
import asyncio
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


async def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )

    try:
        # Agent 1: HR Writer
        hr_writer = AssistantAgent(
            name="hr_writer",
            model_client=model_client,
            system_message=(
                "You are a professional HR assistant. "
                "Create clear, structured, and accurate job "
                "descriptions based on the requested position. "
                "Focus on responsibilities, technical skills, "
                "qualifications, and teamwork."
            )
        )

        # Agent 2: Recruiter
        recruiter = AssistantAgent(
            name="recruiter",
            model_client=model_client,
            system_message=(
                "You are an experienced technical recruiter. "
                "Review job descriptions created by the HR Writer. "
                "Improve readability, candidate appeal, and clarity. "
                "Preserve the original job requirements and avoid "
                "inventing benefits, salary, or company information."
            )
        )

        # Agent 3: HR Reviewer
        hr_reviewer = AssistantAgent(
            name="hr_reviewer",
            model_client=model_client,
            system_message=(
                "You are a senior HR quality reviewer. "
                "Review the Recruiter's revised job description. "
                "Check completeness, professional tone, accuracy, "
                "inclusive language, and formatting. "
                "Return a polished final job description without "
                "adding unsupported requirements."
            )
        )

        print("HR Writer Agent initialized successfully!")
        print("Recruiter Agent initialized successfully!")
        print("HR Reviewer Agent initialized successfully!")
        print("All three AutoGen agents are ready.")

    
        # -----------------------------------
        # STEP 5: Sequential Agent Execution
        # -----------------------------------

        # Agent 1: HR Writer
        print("\n========== AGENT 1: HR WRITER ==========")

        hr_task = (
            "Create a job description for a Junior Python Developer. "
            "Write exactly 5 numbered bullet points. "
            "Include responsibilities, Python skills, frameworks, "
            "qualifications, and teamwork. "
            "Do not invent company details or salary information."
        )

        hr_result = await hr_writer.run(task=hr_task)
        hr_output = hr_result.messages[-1].content

        if not isinstance(hr_output, str) or not hr_output.strip():
            raise ValueError("HR Writer returned no text.")

        print("\nHR Writer Output:\n")
        print(hr_output)

        # Agent 2: Recruiter
        print("\n========== AGENT 2: RECRUITER ==========")

        recruiter_task = (
            "Improve the following Junior Python Developer "
            "job description to attract suitable candidates. "
            "Maintain exactly 5 numbered bullet points. "
            "Preserve the original requirements and avoid "
            "inventing benefits or qualifications.\n\n"
            f"HR Writer Draft:\n{hr_output}"
        )

        recruiter_result = await recruiter.run(
            task=recruiter_task
        )

        recruiter_output = recruiter_result.messages[-1].content

        if not isinstance(recruiter_output, str) or not recruiter_output.strip():
            raise ValueError("Recruiter returned no text.")

        print("\nRecruiter Output:\n")
        print(recruiter_output)

        # Agent 3: HR Reviewer
        print("\n========== AGENT 3: HR REVIEWER ==========")

        reviewer_task = (
            "Review and finalize the following revised job "
            "description for a Junior Python Developer. "
            "Correct unclear wording and check completeness, "
            "professional tone, and inclusive language. "
            "Return exactly 5 numbered bullet points. "
            "Output only the final job description.\n\n"
            f"Recruiter Draft:\n{recruiter_output}"
        )

        reviewer_result = await hr_reviewer.run(
            task=reviewer_task
        )

        final_output = reviewer_result.messages[-1].content

        if not isinstance(final_output, str) or not final_output.strip():
            raise ValueError("HR Reviewer returned no text.")

        print("\n========== FINAL APPROVED DRAFT ==========\n")
        print(final_output)

        print("\nSequential workflow completed successfully!")

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
