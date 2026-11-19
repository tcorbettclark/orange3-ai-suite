"""The feature guide: one page generated from the catalogue."""

from orangecontrib.ai_suite.kernel.images.features import (
    all_features,
    all_groups,
    feature_guide_markdown,
)


def test_guide_covers_every_feature_and_group():
    page = feature_guide_markdown()
    for group in all_groups():
        assert f"## {group.letter} · {group.title}" in page
        assert group.description in page
    for feature in all_features():
        description = feature.description
        assert f"### {description.title}" in page
        assert f"`{feature.name}`" in page
        assert description.intuition in page
        assert description.technical in page
        for reference in description.references:
            assert reference in page


def test_guide_is_markdown_the_widget_can_render():
    page = feature_guide_markdown()
    # Qt's own Markdown renderer has no table support, and neither does this page.
    assert "|" not in page
    assert "```" not in page
    assert page.startswith("# Image features\n")
    assert page.endswith("\n") and not page.endswith("\n\n")


def test_guide_orders_features_by_column():
    page = feature_guide_markdown()
    headings = [page.index(f"### {feature.description.title}") for feature in all_features()]
    assert headings == sorted(headings)
