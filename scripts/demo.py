#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from smolagents import ApiWebSearchTool, OpenAIModel, ToolCallingAgent, VisitWebpageTool

load_dotenv()


def save_trajectory_to_jsonl(trajectory: list[dict], output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Trajectory saved to: {output_path} ({len(trajectory)} steps)")


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
    args = parser.parse_args()

    search_tool = ApiWebSearchTool(rate_limit=1.0)
    browse_tool = VisitWebpageTool(max_output_length=40000)

    model = OpenAIModel(
        model_id=args.model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        tool_choice="auto",
    )

    agent = ToolCallingAgent(
        tools=[search_tool, browse_tool],
        model=model,
        max_steps=args.max_steps,
    )

    print(f"Model: {args.model}")
    print(f"Max steps: {args.max_steps}")
    print(f"Task: {args.query}\n" + "=" * 80)

    result = agent.run(args.query, return_full_result=True)

    print("\n" + "=" * 80)
    print(f"Final Answer:\n{result.output}")
    print("=" * 80)

    # Save with model name in filename
    model_short = args.model.split("/")[-1].split(":")[0]
    save_trajectory_to_jsonl(result.steps, f"outputs/{model_short}_trajectory.json")


if __name__ == "__main__":
    main()
