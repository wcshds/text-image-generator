use image::{imageops::FilterType, GenericImage, GrayImage, Luma};
use imageproc::rect::Rect;
use nalgebra::{Matrix3, Matrix4, Matrix4x2, Matrix4x3};
use numpy::{PyArray, PyArray2, PyReadonlyArray2};
use once_cell::sync::Lazy;
use pyo3::{pyclass, pymethods, types::PyType, Python};
use rand::{
    distributions::{Distribution, Uniform},
    seq::SliceRandom,
    Rng,
};

use super::effect_helper::{
    cv::{self, rectangle},
    gaussian_blur::GaussBlur,
    math::Random,
};

#[inline]
fn get_rotate_matrix(x: f32, y: f32, z: f32) -> Matrix4<f32> {
    let x = x.to_radians();
    let y = y.to_radians();
    let z = z.to_radians();

    let (sin_x, cos_x) = x.sin_cos();
    let (sin_y, cos_y) = y.sin_cos();
    let (sin_z, cos_z) = z.sin_cos();

    #[rustfmt::skip]
    let matrix_x = Matrix4::new(
        1., 0., 0., 0.,
        0., cos_x, -sin_x, 0.,
        0., sin_x, cos_x, 0.,
        0., 0., 0., 1.
    );

    #[rustfmt::skip]
    let matrix_y = Matrix4::new(
        cos_y, 0., sin_y, 0.,
        0., 1., 0., 0.,
        -sin_y, 0., cos_y, 0.,
        0., 0., 0., 1.
    );

    #[rustfmt::skip]
    let matrix_z = Matrix4::new(
        cos_z, -sin_z, 0., 0.,
        sin_z, cos_z, 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.
    );

    matrix_x * matrix_y * matrix_z
}

fn get_warped_pnts(
    points_in: &Matrix4x3<f32>,
    points_out: &Matrix4x3<f32>,
    width: f32,
    height: f32,
    side_length: f32,
) -> (Matrix4x2<f32>, Matrix4x2<f32>) {
    let width_half = width * 0.5;
    let height_half = height * 0.5;
    let side_length_half = side_length * 0.5;

    (
        Matrix4x2::new(
            points_in.m11 + width_half,
            points_in.m12 + height_half,
            points_in.m21 + width_half,
            points_in.m22 + height_half,
            points_in.m31 + width_half,
            points_in.m32 + height_half,
            points_in.m41 + width_half,
            points_in.m42 + height_half,
        ),
        Matrix4x2::new(
            (points_out.m11 + 1.) * side_length_half,
            (points_out.m12 + 1.) * side_length_half,
            (points_out.m21 + 1.) * side_length_half,
            (points_out.m22 + 1.) * side_length_half,
            (points_out.m31 + 1.) * side_length_half,
            (points_out.m32 + 1.) * side_length_half,
            (points_out.m41 + 1.) * side_length_half,
            (points_out.m42 + 1.) * side_length_half,
        ),
    )
}

/// # Parameter:
/// - rotate_angle: (x, y, z)
/// - fovy: field of view along y axis
///
/// # Return:
/// A tutle contains 4 object
/// - 2D transformation matrix
/// - final side length
/// - points input
/// - points output
///
/// # Reference:
/// - [https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles]
/// - [https://nbviewer.org/github/manisoftwartist/perspectiveproj/blob/master/perspective.ipynb]
#[inline]
fn get_warp_matrix(
    width: usize,
    height: usize,
    rotate_angle: (f32, f32, f32),
    scale: f32,
    fovy: f32,
) -> (Matrix3<f32>, f32, Matrix4x2<f32>, Matrix4x2<f32>) {
    let (width, height) = (width as f32, height as f32);
    let (x, y, z) = rotate_angle;

    let fovy_half = ((fovy * 0.5) as f32).to_radians(); // fvHalf
    let distance = (width * width + height * height).sqrt(); // d
    let side_length = scale * distance / fovy_half.cos(); // sideLength
    let hypotenuse = distance / (2.0 * fovy_half.sin()); // h
    let rest = hypotenuse - distance * 0.5; // n
    let full = hypotenuse + distance * 0.5; // f

    let mut translation_mat: Matrix4<f32> = Matrix4::identity();
    translation_mat.m34 = -hypotenuse;

    let rotate_mat: Matrix4<f32> = get_rotate_matrix(x, y, z);

    // initialization should use zeros?
    let mut projection_mat = Matrix4::identity();
    projection_mat.m11 = 1.0 / fovy_half.tan();
    projection_mat.m22 = projection_mat.m11;
    projection_mat.m33 = -(full + rest) / (full - rest);
    projection_mat.m34 = -(2.0 * full * rest) / (full - rest);
    projection_mat.m43 = -1.0;

    let perspective_transform_mat: Matrix4<f32> = projection_mat * translation_mat * rotate_mat;

    let width_half = width * 0.5;
    let height_half = height * 0.5;

    #[rustfmt::skip]
    let points_in: Matrix4x3<f32> = Matrix4x3::new(
        -width_half, height_half, 0.,
        width_half, height_half, 0.,
        width_half, -height_half, 0.,
        -width_half, -height_half, 0.,
    );

    let points_out: Matrix4x3<f32> =
        cv::perspective_transform(&points_in, &perspective_transform_mat);

    let (points_in, points_out) =
        get_warped_pnts(&points_in, &points_out, width, height, side_length);

    (
        cv::get_perspective_transform(&points_in, &points_out),
        side_length,
        points_in,
        points_out,
    )
}

const SHARP_KERNEL: [i32; 9] = [-1, -1, -1, -1, 9, -1, -1, -1, -1]; // 3x3
const EMBOSS_KERNEL: [i32; 9] = [-2, -1, 0, -1, 1, 1, 0, 1, 2]; // 3x3

const UNIFORM_1_2: Lazy<Uniform<f64>> = Lazy::new(|| Uniform::new_inclusive(1.0, 2.0));
const COLOR_50_255: Lazy<Uniform<u8>> = Lazy::new(|| Uniform::new_inclusive(50, 255));
const THICKNESS: [u32; 2] = [1, 2];

#[derive(Clone)]
#[pyclass]
pub struct CvUtil {
    // draw box
    pub box_prob: f64,
    // perspective transform
    pub perspective_prob: f64,
    pub perspective_x: Random,
    pub perspective_y: Random,
    pub perspective_z: Random,
    // gaussian blur
    pub blur_prob: f64,
    pub blur_sigma: Random,
    // filter: emboss/sharp
    pub filter_prob: f64,
    pub emboss_prob: f64,
    pub sharp_prob: f64,
}

impl CvUtil {
    const UNIFORM_0_1: Lazy<Uniform<f64>> = Lazy::new(|| Uniform::new_inclusive(0.0, 1.0));

    pub fn apply_effect(&self, img: GrayImage) -> GrayImage {
        assert!(
            self.emboss_prob + self.sharp_prob == 1.0,
            "emboss probability plus sharp probability should be equal to 1.0"
        );

        let img = if Self::UNIFORM_0_1.sample(&mut rand::thread_rng()) < self.box_prob {
            Self::draw_box(&img, 1.3)
        } else {
            img
        };

        let img = if Self::UNIFORM_0_1.sample(&mut rand::thread_rng()) < self.perspective_prob {
            let rotate_angle = (
                self.perspective_x.sample() as f32,
                self.perspective_y.sample() as f32,
                self.perspective_z.sample() as f32,
            );
            Self::warp_perspective_transform(&img, rotate_angle)
        } else {
            img
        };

        if Self::UNIFORM_0_1.sample(&mut rand::thread_rng()) < self.blur_prob {
            let sigma = self.blur_sigma.sample() as f32;
            let img = Self::gauss_blur(img, sigma);
            if Self::UNIFORM_0_1.sample(&mut rand::thread_rng()) < self.filter_prob {
                if Self::UNIFORM_0_1.sample(&mut rand::thread_rng()) < self.emboss_prob {
                    Self::apply_emboss(&img)
                } else {
                    Self::apply_sharp(&img)
                }
            } else {
                img
            }
        } else {
            img
        }
    }

    /// Perform a perspective transform and crop the transformed text area.
    pub fn warp_perspective_transform(img: &GrayImage, rotate_angle: (f32, f32, f32)) -> GrayImage {
        let (raw_height, raw_width) = (img.height(), img.width());

        let (transform_mat, side_length, _, points_out) = get_warp_matrix(
            raw_width as usize,
            raw_height as usize,
            rotate_angle,
            1.0,
            50.,
        );

        let (raw_height, raw_width) = (raw_height as f32, raw_width as f32);
        let side_length = side_length.ceil() as u32;

        let mut warp_img = cv::warp_perspective(img, &transform_mat, side_length, Luma([0]));

        let (min_x, max_x, min_y, max_y) = (
            points_out.column(0).min(),
            points_out.column(0).max(),
            points_out.column(1).min(),
            points_out.column(1).max(),
        );
        let (min_x, min_y, max_x, max_y) = (
            min_x.floor() as u32,
            min_y.floor() as u32,
            max_x.ceil() as u32,
            max_y.ceil() as u32,
        );
        let crop_img = warp_img
            .sub_image(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image();

        let (new_height, new_width) = (crop_img.height() as f32, crop_img.width() as f32);
        let (resize_width, resize_height) = (
            (new_width * raw_height / new_height).ceil() as u32,
            raw_height as u32,
        );
        let resize_img = if resize_width <= raw_width as u32 && resize_height <= raw_height as u32 {
            image::imageops::resize(&crop_img, resize_width, resize_height, FilterType::Triangle)
        } else {
            let (resize_width, resize_height) = (
                raw_width as u32,
                (new_height * raw_width / new_width).ceil() as u32,
            );
            image::imageops::resize(&crop_img, resize_width, resize_height, FilterType::Triangle)
        };

        resize_img
    }

    pub fn apply_emboss(img: &GrayImage) -> GrayImage {
        let res = imageproc::filter::filter3x3(&img, &EMBOSS_KERNEL);
        res
    }

    pub fn apply_sharp(img: &GrayImage) -> GrayImage {
        let res = imageproc::filter::filter3x3(&img, &SHARP_KERNEL);
        res
    }

    /// Blur the image to simulate the effect of enlarging the small image
    pub fn apply_down_up(img: &GrayImage) -> GrayImage {
        let scale = UNIFORM_1_2.sample(&mut rand::thread_rng());
        let height = img.height();
        let width = img.width();

        let reduced = image::imageops::resize(
            img,
            (width as f64 / scale) as u32,
            (height as f64 / scale) as u32,
            FilterType::Triangle,
        );
        image::imageops::resize(&reduced, width, height, FilterType::Triangle)
    }

    pub fn gauss_blur(img: GrayImage, sigma: f32) -> GrayImage {
        GaussBlur::gaussian_blur(img, sigma, 0.0)
    }

    pub fn draw_box(img: &GrayImage, alpha: f64) -> GrayImage {
        assert!(alpha >= 1.0, "alpha should be greater than 1.0");

        let (height, width) = (img.height(), img.width());
        let (pad_height, pad_width) = (
            (height as f64 * alpha).ceil() as u32,
            (width as f64 * alpha).ceil() as u32,
        );
        let top = rand::thread_rng().gen_range(1..=(pad_height - height));
        let left = rand::thread_rng().gen_range(1..=(pad_width - width));

        let mut img_pad = GrayImage::from_pixel(pad_width, pad_height, Luma([0]));
        img_pad
            .copy_from(img, left, top)
            .expect("origin image is smaller than padded image");

        let box_left = rand::thread_rng().gen_range(1..=(left as i32));
        let box_top = rand::thread_rng().gen_range(1..=(top as i32));
        let box_width = rand::thread_rng()
            .gen_range((width + left - box_left as u32)..=(pad_width - box_left as u32));
        let box_height = rand::thread_rng()
            .gen_range((height + top - box_top as u32)..=(pad_height - box_top as u32));

        let rect = Rect::at(box_left, box_top).of_size(box_width, box_height);
        let color = Luma([COLOR_50_255.sample(&mut rand::thread_rng())]);
        let thickness = THICKNESS.choose(&mut rand::thread_rng()).unwrap().clone();

        rectangle(&mut img_pad, rect, color, thickness);

        image::imageops::resize(&img_pad, width, height, FilterType::Triangle)
    }
}

#[pymethods]
impl CvUtil {
    #[pyo3(name = "apply_effect")]
    pub fn apply_effect_py<'py>(
        &self,
        img: PyReadonlyArray2<'py, u8>,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = img.shape();
        let img = img.as_slice().expect("fail to read input `img`");
        let img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, img.to_vec())
            .expect("fail to cast input img to GrayImage");

        let res = self.apply_effect(img);
        let [height_after, width_after] = [res.height() as usize, res.width() as usize];

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([height_after, width_after]).unwrap();

        reshape_py
    }

    #[classmethod]
    #[pyo3(name = "warp_perspective_transform")]
    pub fn warp_perspective_transform_py<'py>(
        _cls: &PyType,
        img: PyReadonlyArray2<'py, u8>,
        rotate_angle: (f32, f32, f32),
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = img.shape();
        let img = img.as_slice().expect("fail to read input `img`");
        let img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, img.to_vec())
            .expect("fail to cast input img to GrayImage");

        let res = Self::warp_perspective_transform(&img, rotate_angle);
        let [height_after, width_after] = [res.height() as usize, res.width() as usize];

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([height_after, width_after]).unwrap();

        reshape_py
    }

    #[classmethod]
    #[pyo3(name = "apply_emboss")]
    pub fn apply_emboss_py<'py>(
        _cls: &PyType,
        img: PyReadonlyArray2<'py, u8>,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = img.shape();
        let img = img.as_slice().expect("fail to read input `img`");
        let img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, img.to_vec())
            .expect("fail to cast input img to GrayImage");

        let res = Self::apply_emboss(&img);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([shape[0], shape[1]]).unwrap();

        reshape_py
    }

    #[classmethod]
    #[pyo3(name = "apply_sharp")]
    pub fn apply_sharp_py<'py>(
        _cls: &PyType,
        img: PyReadonlyArray2<'py, u8>,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = img.shape();
        let img = img.as_slice().expect("fail to read input `img`");
        let img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, img.to_vec())
            .expect("fail to cast input img to GrayImage");

        let res = Self::apply_sharp(&img);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([shape[0], shape[1]]).unwrap();

        reshape_py
    }

    #[classmethod]
    #[pyo3(name = "apply_down_up")]
    pub fn apply_down_up_py<'py>(
        _cls: &PyType,
        img: PyReadonlyArray2<'py, u8>,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = img.shape();
        let img = img.as_slice().expect("fail to read input `img`");
        let img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, img.to_vec())
            .expect("fail to cast input img to GrayImage");

        let res = Self::apply_down_up(&img);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([shape[0], shape[1]]).unwrap();

        reshape_py
    }

    #[classmethod]
    #[pyo3(name = "gauss_blur")]
    pub fn gauss_blur_py<'py>(
        _cls: &PyType,
        img: PyReadonlyArray2<'py, u8>,
        sigma: f32,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = img.shape();
        let img = img.as_slice().expect("fail to read input `img`");
        let img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, img.to_vec())
            .expect("fail to cast input img to GrayImage");

        let res = Self::gauss_blur(img, sigma);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([shape[0], shape[1]]).unwrap();

        reshape_py
    }

    #[classmethod]
    #[pyo3(name = "draw_box")]
    pub fn draw_box_py<'py>(
        _cls: &PyType,
        img: PyReadonlyArray2<'py, u8>,
        alpha: f64,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = img.shape();
        let img = img.as_slice().expect("fail to read input `img`");
        let img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, img.to_vec())
            .expect("fail to cast input img to GrayImage");

        let res = Self::draw_box(&img, alpha);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([shape[0], shape[1]]).unwrap();

        reshape_py
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use super::*;

    fn create_cv_util() -> CvUtil {
        CvUtil {
            box_prob: 0.1,
            perspective_prob: 0.2,
            perspective_x: Random::new_gaussian(-15.0, 15.0),
            perspective_y: Random::new_gaussian(-15.0, 15.0),
            perspective_z: Random::new_gaussian(-3.0, 3.0),
            blur_prob: 0.1,
            blur_sigma: Random::new_uniform(0.0, 1.5),
            filter_prob: 0.01,
            emboss_prob: 0.4,
            sharp_prob: 0.6,
        }
    }

    #[test]
    fn test_effect() {
        let start = Instant::now();
        let img = image::open("./test-img/test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let cv_util = create_cv_util();
        let res = cv_util.apply_effect(gray);

        res.save("./test-img/cv_effect.png").unwrap();
        println!("cv effect elapsed: {}", start.elapsed().as_secs_f64());
    }

    #[test]
    fn test_warp_perspective_transform() {
        let start = Instant::now();
        let img = image::open("./test-img/test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let res = CvUtil::warp_perspective_transform(&gray, (-3., -3., -3.));

        res.save("./test-img/warp.png").unwrap();
        println!("warp elapsed: {}", start.elapsed().as_secs_f64());
    }

    #[test]
    fn test_sharp() {
        let start = Instant::now();
        let img = image::open("./test-img/test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let res = CvUtil::apply_sharp(&gray);

        res.save("./test-img/sharp.png").unwrap();
        println!("sharp elapsed: {}", start.elapsed().as_secs_f64());
    }

    #[test]
    fn test_emboss() {
        let start = Instant::now();
        let img = image::open("./test-img/test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let res = CvUtil::apply_emboss(&gray);

        res.save("./test-img/emboss.png").unwrap();
        println!("emboss elapsed: {}", start.elapsed().as_secs_f64());
    }

    #[test]
    fn test_down_up() {
        let start = Instant::now();
        let img = image::open("./test-img/test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let res = CvUtil::apply_down_up(&gray);

        res.save("./test-img/down_up.png").unwrap();
        println!("down up elapsed: {}", start.elapsed().as_secs_f64());
    }

    #[test]
    fn test_gauss_blur() {
        let start = Instant::now();
        let img = image::open("./test-img/test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let res = CvUtil::gauss_blur(gray, 1.5);

        res.save("./test-img/gauss_blur.png").unwrap();
        println!("gaussian blur elapsed: {}", start.elapsed().as_secs_f64());
    }

    #[test]
    fn test_draw_box() {
        let start = Instant::now();
        let img = image::open("./test-img/test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let res = CvUtil::draw_box(&gray, 1.3);

        res.save("./test-img/box.png").unwrap();
        println!("draw box elapsed: {}", start.elapsed().as_secs_f64());
    }
}
