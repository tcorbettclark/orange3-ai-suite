import numpy as np

from .._imaging import block_mean_variance
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class CoarseTexture(Feature):
    name = "coarse_texture"
    description = Description(
        title="Coarse texture",
        summary="Variance of the 16x16 block means: large-scale patchiness.",
        intuition=(
            "Whether the picture is made of big soft lumps or is broadly even. Puffy cloud, a "
            "field of boulders seen from far away and mottled terrain score high; a flat haze "
            "or a plain wall scores low. It deliberately ignores fine detail, so grain that "
            "would lift the fine-texture feature does not lift this one."
        ),
        technical=(
            "The image is reduced with INTER_AREA to one value per 16x16 block, and the "
            "variance of those block means is reported. Downsampling by area averages each "
            "block before the variance is taken, so fine-scale noise is suppressed and only "
            "structure larger than the block survives. Pixel-scale in the same sense as "
            "`fine_texture`."
        ),
        references=(
            "[OpenCV: resize](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return block_mean_variance(image)
