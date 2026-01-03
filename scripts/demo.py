#!/usr/bin/env python3
import argparse
import os

from dotenv import load_dotenv
from smolagents import ApiWebSearchTool, OpenAIModel, ToolCallingAgent, VisitWebpageTool

from utils import sanitize_trajectory, save_trajectory

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run a web search agent with different models")
    parser.add_argument(
        "-m", "--model",
        default="minimax/minimax-m2.1",
        help="Model ID to use (e.g., minimax/minimax-m2.1, z-ai/glm-4.7). Default: minimax/minimax-m2.1",
    )
    parser.add_argument(
        "-q", "--query",
        default="Who is the president of the united states as of the end of 2025?",
        help="Query to run",
    )
    parser.add_argument(
        "-n", "--max-steps",
        type=int,
        default=100,
        help="Maximum number of agent steps. Default: 100",
    )
    parser.add_argument(
        "-s", "--system-prompt",
        type=str,
        default=None,
        help="Path to a custom system prompt file (markdown or text)",
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default: 1.0",
    )
    parser.add_argument(
        "-p", "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling. Default: 0.95",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (passed via extra_body for OpenRouter). Default: 40",
    )
    args = parser.parse_args()

    # Load custom system prompt if provided
    custom_instructions = None
    if args.system_prompt:
        with open(args.system_prompt, "r") as f:
            custom_instructions = f.read().strip()

    search_tool = ApiWebSearchTool(rate_limit=1.0)
    browse_tool = VisitWebpageTool(max_output_length=40000)

    model = OpenAIModel(
        model_id=args.model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        tool_choice="auto",
        temperature=args.temperature,
        top_p=args.top_p,
        extra_body={"top_k": args.top_k},
    )

    agent = ToolCallingAgent(
        tools=[search_tool, browse_tool],
        model=model,
        max_steps=args.max_steps,
        instructions=custom_instructions,
    )

    print(f"Model: {args.model}")
    print(f"Max steps: {args.max_steps}")
    print(f"Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"Task: {args.query}\n" + "=" * 80)

    result = agent.run(args.query, return_full_result=True)

    print("\n" + "=" * 80)
    print(f"Final Answer:\n{result.output}")
    print("=" * 80)

    # Sanitize and save trajectory
    model_short = args.model.split("/")[-1].split(":")[0]
    trajectory = sanitize_trajectory(result.steps, agent.tools, agent.system_prompt)
    save_trajectory(trajectory, f"outputs/{model_short}_trajectory.json")


if __name__ == "__main__":
    main()
