from typing import Callable, List

from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from mdit_py_plugins.wordcount import basic_count


def word_count_plugin(
    md: MarkdownIt,
    *,
    per_minute: int = 80,
    count_func: Callable[[str], int] = basic_count,
    store_text: bool = False,
    measured_token_types: tuple[str] = ("text", "fence", "code_block", "html_block"),
    nested_token_types: tuple[str] = ("inline",),
) -> None:
    """
    This is a slightly modified version of the wordcount plugin, that includes code snippets to calculate the reading
    time of the document. The original plugin is available in the `mdit_py_plugins.wordcount` package.
    """

    def _word_count_rule(state: StateCore) -> None:
        text: List[str] = []
        words = 0
        for token in state.tokens:
            if token.type in measured_token_types:
                words += count_func(token.content)
                if store_text:
                    text.append(token.content)
            elif token.type in nested_token_types:
                for child in token.children or ():
                    if child.type in measured_token_types:
                        words += count_func(child.content)
                        if store_text:
                            text.append(child.content)

        data = state.env.setdefault("wordcount", {})
        if store_text:
            data.setdefault("text", [])
            data["text"] += text
        data.setdefault("words", 0)
        data["words"] += words
        data["minutes"] = int(round(data["words"] / per_minute))

    md.core.ruler.push("wordcount", _word_count_rule)
