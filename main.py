import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool, CodeDocsSearchTool

load_dotenv()

llm = LLM(model="gpt-4o-mini")
current_dir = os.path.dirname(os.path.abspath(__file__))

analyzer_agent = Agent(
    role="Python Code Analyzer",
    goal="Analyze documents and learn to implement complex algorithms in Python.",
    backstory="""You are an expert Python developer tasked with reading and analyzing competitive programming documents. 
    The document contains several problem statements and their solutions explained in natural language and math notation. 
    You are responsible for analyzing the steps to solve the problem and to translate what it says into code.""",
    llm=llm,
)

coder_agent = Agent(
    role="Senior Python Developer",
    goal="Generate code solving the problems provided and write it in Python file.",
    backstory="""You are a senior Python developer and competitive programming expert with extensive experience 
    in solving problems that require complex algorithms. 
    You have been tasked with translating the problems solutions into executable Python code.
    The Python script must contain one endpoint for each problem in the document.
    You must also search and read the FastAPI, SciPy and Numpy documentation to help you wrting functions to solve the problems. 
    You will execute it, and ensurance the code does not contain errors.""",
    allow_code_execution=True,
    llm=llm,
)

read_pdf_task = Task(
    description="""
    Read the provided PDF document and analyze the step-by-step solutions described in to translate what it say into code.
    The document contains several problem statements and their solutions explained in natural language and math notation.
    """,
    expected_output="A detailed analysis of the solutions discussed in the PDF document.",
    agent=analyzer_agent,
    tools=[PDFSearchTool(pdf=os.path.join(current_dir, "finals2023solutions.pdf"), )],
)

learn_fastapi_task = Task(
    description="Fetch and analyze the FastAPI documentation.",
    expected_output="A summary of key FastAPI documentation points relevant to the code.",
    agent=coder_agent,
    tools=[CodeDocsSearchTool(docs_url="https://fastapi.tiangolo.com/")],
)

learn_scipy_task = Task(
    description="Fetch and analyze the SciPy documentation.",
    expected_output="A summary of key SciPy documentation points relevant to the code.",
    agent=coder_agent,
    tools=[CodeDocsSearchTool(docs_url="https://docs.scipy.org/doc/scipy/")],
)

learn_numpy_task = Task(
    description="Fetch and analyze the Numpy documentation.",
    expected_output="A summary of key Numpy documentation points relevant to the code.",
    agent=coder_agent,
    tools=[CodeDocsSearchTool(docs_url="https://numpy.org/doc/2.1/")],
)

generate_code_task = Task(
    description="""Generate an executable Python program that solves the problems provided in the document, 
    you can use tools to read the FastAPI, SciPy and Numpy docs only to help you. 
    The code should implement the functionality for each problem described in a different endpoint and be well-documented. 
    Each endpoint should be a post, for example '/A', '/B', etc. 
    The output should be valid Python code, not a description.
    """,
    expected_output="""
    A valid Python '.py' file implementing the solution descriptions provided, 
    with clear comments and code structure. 
    Just the code itself, without any markdown notations around it nor backticks to mark code.""",
    agent=coder_agent,
    output_file="solutions.py"
)

dev_crew = Crew(
    agents=[analyzer_agent, coder_agent],
    tasks=[read_pdf_task, learn_fastapi_task, learn_scipy_task, learn_numpy_task, generate_code_task],
    verbose=True
)

result = dev_crew.kickoff()

print(result)