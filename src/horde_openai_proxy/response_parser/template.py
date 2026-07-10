# Copyright 2026 The HuggingFace Team. All rights reserved.
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
#
# From https://github.com/huggingface/transformers/blob/main/tests/utils/test_chat_parsing.py

__all__ = [
    'cohere_template',
    'ernie_template',
    'gemma4_template',
    'gpt_oss_template',
    'qwen3_template',
    'smollm_template',
]

cohere_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "fields": {
        "content": {
            "open": "<|START_RESPONSE|>",
            "close": "<|END_RESPONSE|>",
            "content": "text",
        },
        "thinking": {
            "open": "<|START_THINKING|>",
            "close": "<|END_THINKING|>",
            "content": "text",
        },
        "tool_calls": {
            "open": "<|START_ACTION|>",
            "close": "<|END_ACTION|>",
            "content": "json",
            "transform_each": True,
            "transform": {
                "type": "function",
                "function": {"name": "{tool_name}", "arguments": "{parameters}"},
            },
        },
    },
}

ernie_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "Assistant:",
    "fields": {
        "thinking": {
            "open_pattern": r"(?:^|<think>\s*)",
            "close": "</think>",
            "content": "text",
        },
        "content": {
            "open": "<response>\n",
            "close_pattern": r"\n?</response>",
            "content": "text",
        },
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
    },
}

gpt_oss_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|start|>assistant",
    "fields": {
        "thinking": {
            "open": "<|channel|>analysis<|message|>",
            "close": "<|end|>",
            "content": "text",
        },
        "content": {
            "open": "<|channel|>final<|message|>",
            "close": "<|end|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>",
            "close": "<|call|>",
            "repeats": True,
            "content": "json",
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}

smollm_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
        "content": {
            "close": "<|im_end|>",
            "content": "text",
        },
    },
}

qwen3_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}

gemma4_template = {
    "defaults": {"role": "assistant"},
    # The chat template only emits `<|turn>model\n` when the previous message wasn't a tool_call/
    # tool_response. After a tool_response the prefix just ends with `<tool_response|>` and the
    # model continues from there, so we accept either anchor and truncate past the latest one.
    "start_anchor": ["<|turn>model\n", "<tool_response|>"],
    "fields": {
        "thinking": {
            "open": "<|channel>thought\n",
            "close": "<channel|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "close": "<tool_call|>",
            "repeats": True,
            "content": "json",
            "content_args": {
                "unquoted_keys": True,
                "string_delims": [['<|"|>', '<|"|>']],
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "close": ["<turn|>", "<|tool_response>", "<eos>"],
            "content": "text",
        },
    },
}
