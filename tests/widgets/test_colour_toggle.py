"""Widget-side colour shortcut toggle (the 40-column demo)."""

from nn_suite.widgets.ow_image_features import OWImageFeatures


def test_widget_colour_toggle_makes_40_columns(qapp, capture, images_table):
    """The shortcut demo: colour group toggled in the widget -> 40 columns."""
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
    from nn_kernel.features import all_feature_names

    assert [v.name for v in out.domain.attributes] == list(all_feature_names())
