import os
from dotenv import load_dotenv
'''
migrating from using gemini 2.5 flash to using Z.ai API with glm-4.5-flash model, high free usage limit.
NOTE: after trial, i have noticed that it has much much higher latancy than Gemini 2.5 flash.
 a single run might take more than 10 seconds
'''
from openai import OpenAI
import argparse

from call_function import available_functions, call_function


ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
MODEL = "glm-4.5-flash"

def main():
    max_loops = 20
    currant_loop = 0
    end_tokenfound = False
    system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Read file contents
- Execute Python files with optional arguments
- Write or overwrite files

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
When finishing a task you should send <task-completed value="true"> to indicate that the task is complete. If you need to perform multiple steps, you can send <task-completed value="false"> until the final step is completed.
the indicator should exist at the end of the response, and should be the only thing in the last line of the response.
the last message should always be a text message, and should not be a function call. If you need to perform multiple steps, you can send <task-completed value="false"> until the final step is completed.
"""
    parser = argparse.ArgumentParser(description="Generate content using the Z.ai API.")
    parser.add_argument("user_prompt", type=str, help="The prompt to send to the Z.ai API.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]

    load_dotenv()
    api_key = os.environ.get("ZAI_API_KEY")

    if api_key is None:
        raise RuntimeError(
            "ZAI_API_KEY not found in environment variables. "
            "Please ensure it is set in your .env file."
        )
    client = OpenAI(api_key=api_key, base_url=ZAI_BASE_URL)
    while currant_loop < max_loops and end_tokenfound == False:
        currant_loop += 1
        #if the last message is reached and it still did not reach a final sloution request another half the limit as it might need it to get the sloution
        if currant_loop == max_loops -1  and end_tokenfound == False:
            print("Warning: Reached maximum loop limit. The task may not be completed.")
            print(f"Requesting additional loops to ensure task completion ({max_loops // 2}), respond with (y)es or (n)o to continue.")
            user_input = input().strip().lower()
            if user_input == "y" or user_input == "yes":
                max_loops = max_loops + max_loops // 2
                print(f"Extended maximum loops to {max_loops}. Continuing...")
            if user_input == "n" or user_input == "no":
                print("Continuing without extending maximum loops. The task may not be completed.")
                

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=available_functions,
        )
    
        if response is None:
            raise RuntimeError(
                "Error trying to generate content. Please check your API key and ensure you have access to the model."
            )
    
        usage = response.usage
        if usage is None:
            raise RuntimeError(
                "There was an issue in accessing the metadata for this response."
                "If there was a response but there were no metadata, contact the app admin."
            )
        if args.verbose:
            print(f"Prompt tokens: {usage.prompt_tokens}\nResponse tokens: {usage.completion_tokens}")
            print(f"User prompt: {args.user_prompt}")
    
        message = response.choices[0].message
        if message.tool_calls:
            #append the tool call to the history, so the agent rememberes what he asked for, and not just the results
            messages.append(message)
            for tool_call in message.tool_calls:
                result_message = call_function(tool_call, args.verbose)
                if not result_message["content"]:
                    raise RuntimeError(f"Function call {tool_call.function.name} returned no content")
                if args.verbose:
                    print(f"-> {result_message['content']}")
                messages.append(result_message)
        else:
            print(message.content)
            messages.append({"role": message.role, "content": message.content})
            end_tokenfound = "<task-completed value=\"true\">" in message.content #a little improvment to reduce the app missing it in case the agent added a white space, or the model hulucinates
    print(f"Total loops: {currant_loop}")


if __name__ == "__main__":
    main()
