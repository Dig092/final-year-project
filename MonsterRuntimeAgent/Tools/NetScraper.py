from openai import OpenAI
import os
import time
import asyncio
import nest_asyncio
from crawl4ai import AsyncWebCrawler
#from duckduckgo_search import AsyncDDGS

from googleapiclient.discovery import build

# Google Custom Search API setup
API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY")
CSE_ID = os.environ.get("GOOGLE_SEARCH_CSE_ID") 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def google_search(query, max_results=5):
    """
    Function to search Google using the Custom Search JSON API.
    """
    retries = 0
    result = ""
    while retries < 3:
        service = build("customsearch", "v1", developerKey=API_KEY)
        result = service.cse().list(q=query, cx=CSE_ID, num=max_results).execute()
        if 'items' not in result:
            retries += 1
            time.sleep(5)
            continue
        return result['items']

    raise RuntimeError(f"Unable to fetch results. Got this result: {result}")

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

async def chat_completion_request(system_prompt,prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt}
        ]
      )
    completion = completion.choices[0].message.content
    return completion

async def generate_search_query(problem_statement):
    """
    Function to call GPT-4 to generate a search query based on a problem statement.
    """

    system_prompt = """
    You are a helpful assistant that can generate 3-5 word search queries from a provided problem statement.
    This search query will be used for crawling helpful resources and documentation to implement code or perform research or analysis of a specific topic of interest.
    As a helpful assistant you have a solid understanding of creating perfect web search queries.
    You also know that you can generate search query with search results for specific websites with this site match pattern `search query site:WEBSITE`.
    Now depending on the type of problem statement you will create a short search query and adding `site:WEBSITE` is optional and not mandatory.
    Some of the helpful websites for specific type of problem statements such as when searching for AI Models documentation, AI models code implementations are:
    - huggingface.co
    - github.com
    Some of the helpful websites for specific type of problem statements such as when searching for Datasets are:
    - huggingface.co
    - github.com
    - kaggle.com

    You can also adjoin multiple websites into a search query if needed by adding "OR" in between the site match patterns.

    Remember:
    - Do not put the query strings inside double quotes.
    """

    prompt = f"Create a single 3-5 word search query for the following problem statement:\n\n{problem_statement}"

    query = await chat_completion_request(system_prompt,prompt)
    print(f"Generated Search Query: {query}")
    return query

# Function to search DuckDuckGo
#async def search_duckduckgo(query, max_results=5):
#    async with AsyncDDGS() as ddgs:
#        results = await ddgs.atext(query, max_results=max_results)
#    return results

# Function to crawl a URL using Crawl4AI
async def crawl_url(url):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
        if result and result.extracted_content:  # Check if content is valid
            return result.extracted_content
        return None

# Function to combine DuckDuckGo search and Crawl4AI crawling
async def search_and_crawl(query, max_results=5):
    search_results = google_search(query, max_results)
    crawled_data = {}
    for result in search_results:
        url = result.get('link')
        if url:
            print(f"Crawling: {url}")
            content = await crawl_url(url)
            if content:
                crawled_data[url] = content
            else:
                print(f"Failed to crawl or extract content from {url}")
    return crawled_data

async def summarize_context(content):
    system_prompt = """
    You are a helpful assistant that is smart to gather accurate context from the provided information and summarizes the information in the most helpful manner.
    As a professional summarizer, create a comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:
    - Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
    - Incorporate main ideas, code implementations, benchmark or capabilities of the particular subject and essential information, eliminating extraneous language and focusing on critical aspects.
    - Rely strictly on the provided text, without including external information.

    If you encounter any type of code in the provided information then make sure to include the code in your summary. The code should be complete. Parse the code properly, structure and indent it properly so it can be used easily in future if needed.

    Conclude your notes with [End of Notes] to indicate completion.
    """
    prompt = f"Please create a summary of this content: {content}"
    completion = await chat_completion_request(system_prompt,prompt)
    return completion

async def retrieve_answer(problem_statement, summarized_content):
    system_prompt = """
    You are a helpful assistant that is able to generate a useful solution for the provided problem statement from the provided context.
    This context can contain implementation code, research analysis, resource links to more information, factual information about events or people or other relevant information.
    There could also be repetitive looking code or information.

    Important point to note:
    You might observe at times that the context is not relevant to the problem and thus the solution you produce out of the context might be incorrect or inappropriate. Avoid hallucination at all costs.
    If you are not confident about the relevance or accuracy of the provided context for the problem statement then just use this for the solution `Unable to retrieve relevant answers`.

    In case of code samples:
    - Use the most relevant information but do not mix and merge code samples if you are not sure of the final working code. Use the most closest solution for code.
    - Also provide source links such as Github repo link etc as additional information along with the code so that we can pull any files that the code might be using. Make sure the source links are correct.
    - Provide instructions along with code for implementation and keep the code in markdown.

    Finally, depending on the type of problem specified you would have to retrieve accurate information from the provided context and generate one final solution.

    Always provide the final solution in a JSON format with the following structure {"solution":"", "confidence_score": INT}.

    Confidence score is your confidence on how relevant and accurate this answer is out of 100%. For calculation of confidence score, perform accuracy match on specific parts of the problem statement with the solution generated by you.
    """
    prompt = f"For this problem: {problem_statement}, generate a relevant solution from this context: {summarized_content}"
    completion = await chat_completion_request(system_prompt,prompt)
    return completion

async def solve_problem_with_search(problem_statement):
    """
    Function that calls GPT-4 API to generate a search query from a problem statement,
    then uses search_and_crawl to get the web content.
    """
    # Step 1: Generate search query from the problem statement
    query = await generate_search_query(problem_statement)

    # Step 2: Perform search and crawl based on the generated query
    crawled_content = await search_and_crawl(query, max_results=3)

    # Step 3: Create a summary for the contents of each crawled URL
    summaries = {}
    final_summary = ""
    if crawled_content!={}:
      for url, content in crawled_content.items():
          if content:
              print(f"Summarizing content from {url}")
              summary = await summarize_context(content)
              summaries[url] = summary
              final_summary = final_summary+f"\n\nSummary for URL: {url}\n{summary}"

    if final_summary!="":
      get_final_answer = asyncio.run(retrieve_answer(problem_statement,final_summary))
    else:
      get_final_answer = "Unable to retrieve relevant answers"

    return crawled_content, summaries, final_summary, get_final_answer


# Example usage
def retreive_from_internet(problem_statement_to_retreive: str):
    #problem_statement = "I need code for running Llama 3.2 11B model"
    crawled_content, summaries, final_summary, get_final_answer = asyncio.run(solve_problem_with_search(problem_statement_to_retreive))
    return get_final_answer
