import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

# --- API KEY CONFIGURATION ---
# Set API keys from Streamlit secrets
# This is a more secure way to handle sensitive information
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    keys_loaded = True
except (KeyError, FileNotFoundError):
    keys_loaded = False

def run_research_agent(company_name, job_role):
    """
    Initializes and runs the LangChain agent to research a company and job role.

    Args:
        company_name (str): The name of the company to research.
        job_role (str): The job role to research.

    Returns:
        str: The research summary from the agent.
    """

    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
    )

    # Initialize the search tool
    # k=5 means it will return the top 5 search results
    search_tool = TavilySearchResults(k=5)
    tools = [search_tool]

    # Pull the ReAct prompt template from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # Create the agent executor to run the agent
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True, # Handles errors if the LLM output is not formatted correctly
        verbose=True  # Shows the agent's thought process in the console
    )

    # Define the detailed input prompt for the agent
    input_prompt = f"""
    Research the company '{company_name}' and the specific job role of '{job_role}'.

    Your final answer MUST be a comprehensive summary structured into two clear sections using Markdown:

    ### **Company Overview**
    * **Domain/Industry**: What is the company's primary domain or industry?
    * **Size**: What is its approximate size (e.g., number of employees)?
    * **Recent News**: Find and summarize one or two recent, significant news articles about the company.

    ### **Role-Specific Requirements**
    * **Common Skills**: What are the most commonly required skills for an '{job_role}' at this company or in the industry?
    * **Experience Level**: What is the typical level of experience (e.g., years, degrees) needed?
    * **Salary Range**: What is the estimated salary range for this role? If a specific range for the company isn't available, provide a general industry estimate.
    """

    # Invoke the agent and get the response
    response = agent_executor.invoke({"input": input_prompt})
    return response['output']

# --- STREAMLIT UI ---
st.title("🤖 AI Research Agent")
st.markdown("This agent uses AI to research a company and a specific job role, providing a detailed summary.")

# Check if API keys are loaded successfully
if not keys_loaded:
    st.error("API keys not found. Please add your GOOGLE_API_KEY and TAVILY_API_KEY to your Streamlit secrets.")
    st.markdown(
        "Create a file named `.streamlit/secrets.toml` in your project directory and add your keys like this:\n"
        "```toml\n"
        "GOOGLE_API_KEY = \"your_google_api_key_here\"\n"
        "TAVILY_API_KEY = \"your_tavily_api_key_here\"\n"
        "```"
    )
else:
    col1, col2 = st.columns(2)

    with col1:
        company_name = st.text_input("Enter Company Name:", placeholder="e.g., Google, Microsoft")

    with col2:
        job_role = st.text_input("Enter Job Role:", placeholder="e.g., Software Engineer, Product Manager")

    if st.button("Start Research", type="primary"):
        if company_name and job_role:
            with st.spinner(f"Researching {company_name} for the role of {job_role}... This may take a moment."):
                try:
                    result = run_research_agent(company_name, job_role)
                    st.markdown("---")
                    st.subheader("Research Summary")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"An error occurred during the research process: {e}")
        else:
            st.warning("Please enter both a company name and a job role.")
