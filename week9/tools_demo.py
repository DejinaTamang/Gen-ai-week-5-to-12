
"""
Week 9: Let Groq use the tools in tools.py.

How the loop works:
1. Send the question and tool list to Groq.
2. If Groq requests a tool, execute it.
3. Send the tool result back to Groq.
4. Continue until Groq gives a normal answer.
"""

import json
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the main .env file
load_dotenv(BASE_DIR / ".env")


from week12.tools import TOOLS, run_tool

MODEL = MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 5


def ask(client, question):
    messages = [
        {"role": "user", "content": question}
    ]

    for step in range(MAX_STEPS):

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=500,
        )

        message = response.choices[0].message

        # No tool requested: return the final answer
        if not message.tool_calls:
            return message.content or ""

        # Save Groq's assistant message
        messages.append(message)

        # Execute requested tools
        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            tool_args = json.loads(
                tool_call.function.arguments
            )

            print(f"  -> Groq called {tool_name} with {tool_args}")

            output = run_tool(tool_name, tool_args)

            print(f"  <- result: {output}")

            # Send tool result back to Groq
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(output),
            })

    return "Stopped: too many tool steps."


def main():

    # Reads GROQ_API_KEY from environment
    client = Groq()

    questions = [
        "What is (12 * 4) + 7?",
        "How long do I have to return an item?",
        "What is 15% of 240, and when is support open?",
        "What is the capital of France?",
    ]

    for q in questions:

        print("=" * 70)
        print(f"Q: {q}")

        answer = ask(client, q)

        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
