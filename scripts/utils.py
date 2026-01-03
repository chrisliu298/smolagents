"""Utility functions for smolagents trajectory processing."""

import json
from pathlib import Path

from smolagents.models import get_tool_json_schema


def sanitize_trajectory(steps: list, tools: dict, system_prompt: str = None) -> dict:
    """
    Convert smolagents trajectory to clean ChatML format compatible with
    tokenizer.apply_chat_template(messages, tools=tools).

    Steps are dicts (serialized from TaskStep/ActionStep dataclasses).

    Args:
        steps: List of step dicts from agent.run(return_full_result=True).steps
        tools: Dict of tool name -> Tool from agent.tools
        system_prompt: Optional system prompt string

    Returns:
        Dict with 'tools' (list of tool schemas) and 'messages' (list of ChatML messages)
    """
    # Extract tool schemas in OpenAI format (tools is a dict with name -> Tool)
    tool_schemas = [get_tool_json_schema(tool) for tool in tools.values()]

    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

    for step in steps:
        # Handle TaskStep (has 'task' key, no 'step_number')
        if "task" in step and "step_number" not in step:
            messages.append({
                "role": "user",
                "content": f"New task:\n{step['task']}",
            })
            continue

        # Handle PlanningStep (has 'plan' key)
        if "plan" in step:
            messages.append({
                "role": "assistant",
                "content": step["plan"].strip(),
            })
            messages.append({
                "role": "user",
                "content": "Now proceed and carry out this plan.",
            })
            continue

        # Handle ActionStep (has 'step_number' key)
        if "step_number" in step:
            # Extract reasoning from raw response if available
            reasoning = None
            model_output_msg = step.get("model_output_message")
            if model_output_msg:
                raw = model_output_msg.get("raw")
                if raw and "choices" in raw and raw["choices"]:
                    raw_msg = raw["choices"][0].get("message", {})
                    reasoning = raw_msg.get("reasoning")

            # Build assistant message
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                # Assistant message with tool calls
                tool_calls_formatted = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls_formatted.append({
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": func.get("name"),
                            "arguments": args,
                        }
                    })

                assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_formatted,
                }
                if reasoning:
                    assistant_msg["reasoning"] = reasoning
                messages.append(assistant_msg)

                # Tool response messages (observations)
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name")
                    tool_call_id = tc.get("id")

                    if tool_name == "final_answer":
                        action_output = step.get("action_output")
                        content = str(action_output) if action_output is not None else ""
                    else:
                        # Add "Observation:\n" prefix to match memory.py:130-140
                        obs = step.get("observations") or ""
                        content = f"Observation:\n{obs}" if obs else ""

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    })

                # Error feedback (separate from observations, matches memory.py:142-152)
                error = step.get("error")
                if error and tool_calls:
                    error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                    error_content = (
                        f"Call id: {tool_calls[0].get('id')}\n"
                        f"Error:\n{error_msg}\n"
                        "Now let's retry: take care not to repeat previous errors! "
                        "If you have retried several times, try a completely different approach.\n"
                    )
                    # Error feedback as user message (no tool_call_id available for error)
                    messages.append({
                        "role": "user",
                        "content": error_content,
                    })
            elif step.get("model_output"):
                # Assistant message without tool calls (text response)
                content = step["model_output"]
                if isinstance(content, list):
                    content = content[0].get("text", "") if content else ""

                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                }
                if reasoning:
                    assistant_msg["reasoning"] = reasoning
                messages.append(assistant_msg)

                # If there's an error but no tool_calls (e.g., parsing failed),
                # add error as a user message (matches memory.py behavior)
                error = step.get("error")
                if error:
                    error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                    error_content = (
                        f"Error:\n{error_msg}\n"
                        "Now let's retry: take care not to repeat previous errors! "
                        "If you have retried several times, try a completely different approach.\n"
                    )
                    messages.append({
                        "role": "user",
                        "content": error_content,
                    })

    return {
        "tools": tool_schemas,
        "messages": messages,
    }


def save_trajectory(data: dict, output_file: str):
    """Save sanitized trajectory to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Trajectory saved to: {output_path} ({len(data['messages'])} messages)")
