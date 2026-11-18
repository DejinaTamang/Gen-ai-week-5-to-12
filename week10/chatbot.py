
"""
Week 10: A chatbot built with LangChain and Groq.

Framework parts used here:
- ChatGroq: the model connection
- ChatPromptTemplate: a reusable prompt with a slot for history
- RunnableWithMessageHistory: stores and replays the conversation
- /clear command: clears conversation memory
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


# Load the main .env file from GENAI 5
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# Check API key
if not os.environ.get("GROQ_API_KEY"):
    sys.exit(
        "GROQ_API_KEY is not set. "
        "Please check your main .env file."
    )


SESSION_ID = "default"


# Connect to Groq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    max_tokens=500,
    temperature=0
)


# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly assistant for an online store. "
        "Keep answers short and clear."
    ),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])


# Create chain
chain = prompt | llm


# Conversation memory
store: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:

    return store.setdefault(
        session_id,
        InMemoryChatMessageHistory()
    )


# Add conversation history to chatbot
chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)


def main() -> None:

    print(
        "Store chatbot (LangChain + Groq version).\n"
        "Type /clear to forget the conversation, "
        "or quit to exit.\n"
    )

    while True:

        try:
            user_input = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":

            store.pop(SESSION_ID, None)

            print("Memory cleared.\n")
            continue

        try:

            reply = chatbot.invoke(
                {"input": user_input},
                config={
                    "configurable": {
                        "session_id": SESSION_ID
                    }
                },
            )

            print(f"Bot: {reply.content}\n")

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
