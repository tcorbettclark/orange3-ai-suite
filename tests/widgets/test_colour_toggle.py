"""Widget-side colour shortcut toggle (the full 38-column demo)."""

from orangecontrib.ai_suite.widgets.ow_image_features import OWImageFeatures


def test_widget_colour_toggle_adds_colour_columns(qapp, capture, images_table):
    """Toggling the colour group in the widget adds all 38 columns."""
    w = OWImageFeatures()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.wait_for_extraction()
    w._group_checks["G"].setChecked(True)
    assert w.groups_enabled == "ABCDEFG"
    w.commit.now()
    w.wait_for_extraction()
    out = features_sent[-1]
    from orangecontrib.ai_suite.kernel.images.features import all_feature_names

    assert [v.name for v in out.domain.attributes] == list(all_feature_names())
