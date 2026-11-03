"""Widget-side colour-trap toggle + scale option (the 40-column demo)."""

import numpy as np

from nn_suite.widgets.ow_image_features import OWImageFeatureExtractor


def test_widget_colour_toggle_makes_40_columns(qapp, capture, images_table):
    """The trap demo: colour group toggled in the widget -> 40 columns."""
    w = OWImageFeatureExtractor()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w._group_checks["G"].setChecked(True)
    assert w.groups_enabled == "ABCDEFG"
    w.commit.now()
    out = features_sent[-1]
    from nn_kernel.features import all_feature_names

    assert [v.name for v in out.domain.attributes] == list(all_feature_names())


def test_widget_scale_option(qapp, capture, images_table):
    w = OWImageFeatureExtractor()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.scale_features = True
    w.commit.now()
    out = features_sent[-1]
    variable_scales = out.X.std(axis=0)
    nonzero = variable_scales[variable_scales > 0]
    # z-scored columns have unit std (zero-variance columns pass through)
    assert np.allclose(nonzero, 1.0)