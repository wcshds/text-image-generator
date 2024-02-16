from typing import Tuple
import numpy.typing as npt

class CvUtil:
    def apply_effect(
        self,
        img: npt.NDArray,
    ) -> npt.NDArray:
        """
        Randomly apply CV effect according to the probs in the config file.

        :param img: grayscale image
        :return: the resulting image
        """
    @classmethod
    def warp_perspective_transform(
        cls, img: npt.NDArray, rotate_angle: Tuple[int, int, int]
    ) -> npt.NDArray:
        """
        Apply warp perspective transform.

        :param img: grayscale image
        :param rotate_angle: rotate angles (x, y, z)
        :return: the resulting image
        """
    @classmethod
    def apply_emboss(cls, img: npt.NDArray) -> npt.NDArray:
        """
        Apply emboss filter.

        :param img: grayscale image
        :return: the resulting image
        """
    @classmethod
    def apply_sharp(cls, img: npt.NDArray) -> npt.NDArray:
        """
        Apply sharp filter.

        :param img: grayscale image
        :return: the resulting image
        """
    @classmethod
    def apply_down_up(cls, img: npt.NDArray) -> npt.NDArray:
        """
        The image is downsampled and then upsampled back to the original size.

        :param img: grayscale image
        :return: the resulting image
        """
    @classmethod
    def gauss_blur(cls, img: npt.NDArray, sigma: float) -> npt.NDArray:
        """
        Gaussian blur is applied to the image.

        :param img: grayscale image
        :param sigma: sigma_x used in gaussian blur
        :return: the resulting image
        """
    @classmethod
    def draw_box(cls, img: npt.NDArray, alpha: float) -> npt.NDArray:
        """
        Put a box border around the image.

        :param img: grayscale image
        :param alpha: zoom factor
        :return: the resulting image
        """

class MergeUtil:
    def random_pad(
        self, font_img: npt.NDArray, bg_height: int, bg_width: int
    ) -> npt.NDArray:
        """
        Randomly reduce the image height by 2 to height_diff pixels while maintaining the aspect ratio, and then pad the image to the specified height and width.

        :param font_img: grayscale text image
        :param bg_height: height of the background image
        :param bg_width: width of the background image
        :return: the resulting image
        """
    def random_change_bgcolor(self, bg_img: npt.NDArray) -> npt.NDArray:
        """
        Randomly change background color.

        :param bg_img: grayscale background image
        :return: the resulting background image
        """
    def poisson_edit(self, font_img: npt.NDArray, bg_img: npt.NDArray) -> npt.NDArray:
        """
        Use poisson editing to merge the text image and the background image.

        :param font_img: grayscale text image
        :param bg_img: grayscale background image
        :return: the resulting merge image
        """

class BgFactory:
    height: int
    width: int

    def __init__(self, dir: str, height: int, width: int) -> None: ...
    def random(self) -> npt.NDArray:
        """
        Get a random background image.

        :return: the resulting background image
        """

class Generator:
    cv_util: CvUtil
    merge_util: MergeUtil
    bg_factory: BgFactory
    font_list: Tuple[str, int, int, int]
    chinese_ch_dict: dict[str, list[Tuple[str, int, int, int]]]
    latin_corpus: str
    latin_ch_dict: dict[str, list[Tuple[str, int, int, int]]]
    symbol_dict: dict[str, list[Tuple[str, int, int, int]]]
    main_font_list: str

    def __init__(self, config_path: str) -> None: ...
    def set_bg_size(
        self,
        height: int,
        width: int,
    ):
        """Set the background image's height and width.

        :param height: specify the height of the background image
        :param width: specify the width of the background image
        """
    def get_random_chinese(
        self, min: int, max: int, add_extra_symbol: bool = False
    ) -> list[Tuple[str, list[Tuple[str, int, int, int]]]]:
        """
        Generate random text with chinese characters.

        :param min: specify the minimum word count for generated text
        :param max: specify the maximum word count for generated text
        :param add_extra_symbol: whether to add punctuation to the generated text
        :return: a list of tuples that contains text and font infos
        """
    def wrap_text_with_font_list(
        self, text: str
    ) -> list[Tuple[str, list[Tuple[str, int, int, int]]]]:
        """
        Gets the available font information for each character in the specified text.

        :param text: a simple sentence of text
        :return: a list of tuples that contains text and font infos
        """
    def gen_image_from_text_with_font_list(
        self,
        text_with_font_list: list[Tuple[str, list[Tuple[str, int, int, int]]]],
        text_color: Tuple[int, int, int],
        background_color: Tuple[int, int, int],
        apply_effect: bool = False,
    ) -> npt.NDArray:
        """
        Generate an image based on a given list of characters and font information.

        :param text_with_font_list: a list of tuples that contains text and font infos
        :param text_color: text color in RGB form
        :param background_color: background color in RGB form
        :param apply_effect: whether to perform image enhancement, if true, the resulting image is a grayscale image
        :return: the resulting image
        """
