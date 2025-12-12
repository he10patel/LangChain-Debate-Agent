from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
