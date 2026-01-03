#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Async tool implementations for smolagents.

These tools use async I/O internally for better performance, wrapped in sync
interfaces for compatibility with smolagents' synchronous execution model.

Example:
    ```python
    from smolagents import AsyncWebSearchTool, AsyncVisitWebpageTool, AsyncBashTool

    search_tool = AsyncWebSearchTool(rate_limit=1.0)
    browse_tool = AsyncVisitWebpageTool(max_output_length=40000)
    bash_tool = AsyncBashTool(timeout=30)
    ```
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)
from .tools import Tool


class AsyncWebSearchTool(Tool):
    """Async web search tool using Brave Search API.

    This tool uses aiohttp for non-blocking HTTP requests.

    Args:
        endpoint (`str`): API endpoint URL. Defaults to Brave Search API.
        api_key (`str`): API key for authentication.
        api_key_name (`str`): Environment variable name containing the API key.
            Defaults to "BRAVE_API_KEY".
        headers (`dict`, *optional*): Headers for API requests.
        params (`dict`, *optional*): Parameters for API requests.
        rate_limit (`float`, default `1.0`): Maximum queries per second.
            Set to `None` to disable rate limiting.
    """

    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, URLs, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        api_key_name: str = "",
        headers: dict = None,
        params: dict = None,
        rate_limit: float | None = 1.0,
    ):
        import os

        super().__init__()
        self.endpoint = endpoint or "https://api.search.brave.com/res/v1/web/search"
        self.api_key_name = api_key_name or "BRAVE_API_KEY"
        self.api_key = api_key or os.getenv(self.api_key_name)
        self.headers = headers or {"X-Subscription-Token": self.api_key}
        self.params = params or {"count": 10}
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

    async def _async_forward(self, query: str) -> str:
        import time

        try:
            import aiohttp
        except ImportError as e:
            raise ImportError(
                "You must install package `aiohttp` to use async tools: run `pip install aiohttp`."
            ) from e

        # Rate limiting
        if self.rate_limit:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()

        params = {**self.params, "q": query}
        async with aiohttp.ClientSession() as session:
            async with session.get(self.endpoint, headers=self.headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()

        results = self._extract_results(data)
        return self._format_markdown(results)

    def _extract_results(self, data: dict) -> list:
        results = []
        for result in data.get("web", {}).get("results", []):
            results.append({
                "title": result["title"],
                "url": result["url"],
                "description": result.get("description", "")
            })
        return results

    def _format_markdown(self, results: list) -> str:
        if not results:
            return "No results found."
        return "## Search Results\n\n" + "\n\n".join([
            f"{idx}. [{result['title']}]({result['url']})\n{result['description']}"
            for idx, result in enumerate(results, start=1)
        ])

    def forward(self, query: str) -> str:
        return asyncio.run(self._async_forward(query))


class AsyncVisitWebpageTool(Tool):
    """Async webpage visitor tool.

    This tool uses aiohttp for non-blocking HTTP requests and converts
    HTML content to markdown.

    Args:
        max_output_length (`int`, default `40000`): Maximum length of the output content.
        timeout (`int`, default `20`): Request timeout in seconds.
    """

    name = "visit_webpage"
    description = "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    inputs = {"url": {"type": "string", "description": "The url of the webpage to visit."}}
    output_type = "string"

    def __init__(self, max_output_length: int = 40000, timeout: int = 20):
        super().__init__()
        self.max_output_length = max_output_length
        self.timeout = timeout

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return content[:max_length] + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"

    async def _async_forward(self, url: str) -> str:
        try:
            import aiohttp
            from markdownify import markdownify
        except ImportError as e:
            raise ImportError(
                "You must install packages `aiohttp` and `markdownify` to use this tool: "
                "run `pip install aiohttp markdownify`."
            ) from e

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()

            # Convert HTML to Markdown
            markdown_content = markdownify(html).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            return self._truncate_content(markdown_content, self.max_output_length)

        except asyncio.TimeoutError:
            return "The request timed out. Please try again later or check the URL."
        except Exception as e:
            return f"Error fetching the webpage: {str(e)}"

    def forward(self, url: str) -> str:
        return asyncio.run(self._async_forward(url))


class AsyncPythonInterpreterTool(Tool):
    """Python interpreter tool that runs code in a thread pool for non-blocking execution.

    This tool wraps the standard Python interpreter in a ThreadPoolExecutor
    to prevent blocking the event loop during CPU-bound code execution.

    Args:
        authorized_imports (`list`, *optional*): List of additional modules to allow importing.
        max_workers (`int`, default `1`): Number of worker threads in the pool.
    """

    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {"code": {"type": "string", "description": "The python code to run in interpreter"}}
    output_type = "string"

    def __init__(self, authorized_imports: list = None, max_workers: int = 1):
        super().__init__()
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))

        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {self.authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _execute_code(self, code: str) -> str:
        state = {}
        output = str(
            self.python_evaluator(
                code,
                state=state,
                static_tools=self.base_python_tools,
                authorized_imports=self.authorized_imports,
            )[0]
        )
        return f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"

    async def _async_forward(self, code: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._execute_code, code)

    def forward(self, code: str) -> str:
        return asyncio.run(self._async_forward(code))


class AsyncBashTool(Tool):
    """Async bash command execution tool.

    This tool uses asyncio.create_subprocess_shell for non-blocking
    command execution with timeout support.

    Args:
        timeout (`int`, default `30`): Command timeout in seconds.
    """

    name = "bash"
    description = "Execute a bash command and return the output. Use this for file operations, system commands, or running scripts."
    inputs = {"command": {"type": "string", "description": "The bash command to execute"}}
    output_type = "string"

    def __init__(self, timeout: int = 30):
        super().__init__()
        self.timeout = timeout

    async def _async_forward(self, command: str) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Error: Command timed out after {self.timeout} seconds"

            stdout_str = stdout.decode().strip()
            stderr_str = stderr.decode().strip()

            output = ""
            if stdout_str:
                output += f"Stdout:\n{stdout_str}\n"
            if stderr_str:
                output += f"Stderr:\n{stderr_str}\n"
            if proc.returncode != 0:
                output += f"Return code: {proc.returncode}"

            return output.strip() if output else "(no output)"

        except Exception as e:
            return f"Error: {str(e)}"

    def forward(self, command: str) -> str:
        return asyncio.run(self._async_forward(command))


async def run_tools_concurrently(tools_and_args: list[tuple[Tool, dict]]) -> list[Any]:
    """Run multiple async tool calls concurrently.

    This utility function allows batch execution of multiple tool calls
    in parallel, improving throughput for independent operations.

    Args:
        tools_and_args: List of (tool, kwargs) tuples where each tool
            will be called with its corresponding keyword arguments.

    Returns:
        List of results in the same order as inputs.

    Example:
        ```python
        results = asyncio.run(run_tools_concurrently([
            (search_tool, {"query": "python async"}),
            (search_tool, {"query": "aiohttp tutorial"}),
            (bash_tool, {"command": "ls -la"}),
        ]))
        ```
    """
    tasks = []
    for tool, kwargs in tools_and_args:
        if hasattr(tool, '_async_forward'):
            tasks.append(tool._async_forward(**kwargs))
        else:
            # Fallback to sync execution in thread pool
            loop = asyncio.get_event_loop()
            tasks.append(loop.run_in_executor(None, lambda t=tool, k=kwargs: t.forward(**k)))

    return await asyncio.gather(*tasks)


__all__ = [
    "AsyncWebSearchTool",
    "AsyncVisitWebpageTool",
    "AsyncPythonInterpreterTool",
    "AsyncBashTool",
    "run_tools_concurrently",
]
