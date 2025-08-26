from openai import OpenAI
import traceback
import time

client = OpenAI()

def talk_with_expert(problem_being_faced: str, max_retries:int = 10) -> str:
    retries = 0
    max_retries = 10
    while retries < max_retries:
        retries += 1
        try:
            refined_prompt = f"""
            Consider that you are a advanced ML engineering expert and below is the problem being faced by a team of ML engineer. 

            Problem: {problem_being_faced}

            Now suggest a proper way to solve this problem.
            """
            completion = client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {"role": "user", "content": refined_prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            traceback.print_exc()
            time.sleep(0.2)
        