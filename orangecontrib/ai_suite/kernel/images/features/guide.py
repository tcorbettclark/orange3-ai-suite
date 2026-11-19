"""The feature guide: every feature's description as one Markdown page.

The page is generated from the catalogue each time it is asked for. The Markdown
stays inside what Qt's own renderer supports - headings, emphasis, inline code,
links and lists.
"""

from __future__ import annotations

from .catalogue import all_features
from .groups import all_groups, group_of

__all__ = ["feature_guide_markdown"]


def feature_guide_markdown() -> str:
    """One Markdown page describing every feature, group by group."""
    lines = [
        "# Image features",
        "",
        "Each feature turns an image into one number. The values are ordinary table",
        "columns, so they can be inspected, plotted and classified like any other data.",
        "Every feature lists what it responds to, and what leaves it cold.",
        "",
    ]
    for group in all_groups():
        lines += [f"## {group.letter} · {group.title}", "", group.description, ""]
        for feature in all_features():
            if group_of(feature.name) != group.letter:
                continue
            description = feature.description
            lines += [
                f"### {description.title}",
                "",
                f"`{feature.name}`",
                "",
                description.summary,
                "",
            ]
            lines += [description.intuition, ""]
            lines += [f"**How it is computed.** {description.technical}", ""]
            if description.references:
                lines += ["**Further reading**", ""]
                lines += [f"- {reference}" for reference in description.references]
                lines += [""]
    return "\n".join(lines).rstrip() + "\n"
