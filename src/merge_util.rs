use std::{fs, ops::Index, path::Path};

use image::{GenericImage, GrayImage, Luma};
use numpy::{PyArray, PyArray2, PyReadonlyArray2};
use pyo3::{pyclass, pymethods, Python};
use rand::Rng;

use super::effect_helper::{
    math::Random,
    poisson_editing::{Gradient, Processor},
};

#[derive(Clone)]
#[pyclass]
pub struct BgFactory {
    images: Vec<GrayImage>,
    pub height: usize,
    pub width: usize,
    pub bg_dir: String,
}

impl BgFactory {
    pub fn new<P: AsRef<Path>>(dir: P, height: usize, width: usize) -> Self {
        let dir_list = fs::read_dir(&dir).expect("background images' directory does not exist");
        let mut image_paths = vec![];

        for each_file in dir_list {
            let each_file = each_file.unwrap().path();
            let extension = match each_file.extension() {
                Some(ext) => ext,
                None => continue,
            };
            if extension == "png" || extension == "jpg" || extension == "jpeg" {
                image_paths.push(each_file)
            }
        }

        let mut images = Vec::with_capacity(image_paths.len());
        for image_path in image_paths {
            let img = match image::open(image_path) {
                Ok(img) => img,
                Err(_) => continue,
            };
            let mut gray = image::imageops::grayscale(&img);

            let [origin_height, origin_width] = [gray.height(), gray.width()];
            if origin_width < width as u32 || origin_height < height as u32 {
                let [width1, height1] = [
                    (origin_width as f64 * height as f64 / origin_height as f64).ceil() as u32,
                    height as u32,
                ];
                let [width2, height2] = [
                    width as u32,
                    (origin_height as f64 * width as f64 / origin_width as f64).ceil() as u32,
                ];
                if width1 >= width as u32 && height1 >= width as u32 {
                    gray = image::imageops::resize(
                        &gray,
                        width1,
                        height1,
                        image::imageops::FilterType::CatmullRom,
                    );
                } else {
                    gray = image::imageops::resize(
                        &gray,
                        width2,
                        height2,
                        image::imageops::FilterType::CatmullRom,
                    );
                }
            }

            // random crop
            let [resize_height, resize_width] = [gray.height(), gray.width()];
            let x = rand::thread_rng().gen_range(0..=(resize_width - width as u32));
            let y = rand::thread_rng().gen_range(0..=(resize_height - height as u32));
            let cropped = gray.sub_image(x, y, width as u32, height as u32).to_image();

            images.push(cropped)
        }

        if images.len() == 0 {
            panic!("No background image exists");
        }

        Self {
            images,
            height,
            width,
            bg_dir: dir.as_ref().to_string_lossy().to_string(),
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn random(&self) -> &GrayImage {
        let index = rand::thread_rng().gen_range(0..self.len());
        &self[index]
    }
}

impl Index<usize> for BgFactory {
    type Output = GrayImage;

    fn index(&self, index: usize) -> &Self::Output {
        self.images.get(index).expect(&format!(
            "index out of range: current index: {}, but total length is {}",
            index,
            self.len()
        ))
    }
}

#[pymethods]
impl BgFactory {
    #[new]
    pub fn py_new(dir: &str, height: usize, width: usize) -> Self {
        let res = Self::new(dir, height, width);
        res
    }

    #[pyo3(name = "__len__")]
    pub fn py_len(&self) -> usize {
        self.len()
    }

    #[getter]
    #[pyo3(name = "width")]
    pub fn py_width(&self) -> usize {
        self.width()
    }

    #[getter]
    #[pyo3(name = "height")]
    pub fn py_height(&self) -> usize {
        self.height()
    }

    #[pyo3(name = "__getitem__")]
    pub fn py_get<'py>(&self, index: usize, _py: Python<'py>) -> &'py PyArray2<u8> {
        let res = &self[index];

        let res_py = PyArray::from_vec(_py, res.to_vec());
        let reshape_py = res_py.reshape([self.height(), self.width()]).unwrap();

        reshape_py
    }

    #[pyo3(name = "random")]
    pub fn py_random<'py>(&self, _py: Python<'py>) -> &'py PyArray2<u8> {
        let res = self.random();

        let res_py = PyArray::from_vec(_py, res.to_vec());
        let reshape_py = res_py.reshape([self.height(), self.width()]).unwrap();

        reshape_py
    }
}

#[derive(Clone)]
#[pyclass]
pub struct MergeUtil {
    pub height_diff: Random,
    pub bg_alpha: Random,
    pub bg_beta: Random,
    pub font_alpha: Random,
    pub reverse_prob: f64,
}

impl MergeUtil {
    fn random_range_u32(a: u32, b: u32) -> u32 {
        if a >= b {
            rand::thread_rng().gen_range(b..=a)
        } else {
            rand::thread_rng().gen_range(a..=b)
        }
    }

    /// bg_shape: (height, width)
    pub fn random_pad(&self, font_img: &GrayImage, bg_height: u32, bg_width: u32) -> GrayImage {
        let (font_height, font_width) = (font_img.height(), font_img.width());

        let resize_height = (bg_height as f64 - self.height_diff.sample()) as u32;
        let resize_width = ((font_width as f64 * resize_height as f64 / font_height as f64) as u32)
            .clamp(1, bg_width);

        let font_img = image::imageops::resize(
            font_img,
            resize_width,
            resize_height,
            image::imageops::FilterType::CatmullRom,
        );

        let top = Self::random_range_u32(1, bg_height - resize_height);
        let left = Self::random_range_u32(0, bg_width - resize_width);

        let mut padded_img = GrayImage::from_pixel(bg_width, bg_height, Luma([0]));
        padded_img.copy_from(&font_img, left, top).unwrap();

        padded_img
    }

    pub fn random_change_bgcolor(&self, bg_img: &GrayImage) -> GrayImage {
        let alpha = self.bg_alpha.sample();
        let beta = self.bg_beta.sample();
        let [width, height] = [bg_img.width(), bg_img.height()];
        let new_bg_img_vec: Vec<_> = bg_img
            .to_vec()
            .iter()
            .map(|&each| ((each as f64 * alpha + beta) as u32).clamp(50, 255) as u8)
            .collect();

        GrayImage::from_vec(width, height, new_bg_img_vec).unwrap()
    }

    pub fn poisson_edit(&self, font_img: &GrayImage, bg_img: &GrayImage) -> GrayImage {
        let bg_img = self.random_change_bgcolor(bg_img);
        let padded_font_img = self.random_pad(&font_img, bg_img.height(), bg_img.width());

        let alpha = self.font_alpha.sample();
        let reversed_adjust_font_img = GrayImage::from_raw(
            padded_font_img.width(),
            padded_font_img.height(),
            padded_font_img
                .pixels()
                .map(|each| {
                    let reversed = (255 - each.0[0]) as f64;
                    let adjust = reversed * alpha;

                    adjust as u8
                })
                .collect(),
        )
        .unwrap();
        let mut poisson_processor = Processor::reset(
            reversed_adjust_font_img,
            padded_font_img,
            bg_img,
            (0, 0),
            (0, 0),
            Gradient::Maximum,
        );
        let (target, _) = poisson_processor.step(500);
        let mut final_img = GrayImage::from_vec(
            target.ncols() as u32,
            target.nrows() as u32,
            target.transpose().iter().map(|&each| each).collect(),
        )
        .unwrap();

        if rand::thread_rng().gen_range(0.0..=1.0) < self.reverse_prob {
            final_img = GrayImage::from_vec(
                final_img.width(),
                final_img.height(),
                final_img.to_vec().iter().map(|each| 255 - each).collect(),
            )
            .unwrap()
        }

        final_img
    }
}

#[pymethods]
impl MergeUtil {
    #[pyo3(name = "random_pad")]
    pub fn random_pad_py<'py>(
        &self,
        font_img: PyReadonlyArray2<'py, u8>,
        bg_height: u32,
        bg_width: u32,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = font_img.shape();
        let font_img = font_img.as_slice().expect("fail to read input `font_img`");
        let font_img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, font_img.to_vec())
            .expect("fail to cast input font_img to GrayImage");

        let res = self.random_pad(&font_img, bg_height, bg_width);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py
            .reshape([bg_height as usize, bg_width as usize])
            .unwrap();

        reshape_py
    }

    #[pyo3(name = "random_change_bgcolor")]
    pub fn random_change_bgcolor_py<'py>(
        &self,
        bg_img: PyReadonlyArray2<'py, u8>,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape = bg_img.shape();
        let bg_img = bg_img.as_slice().expect("fail to read input `bg_img`");
        let bg_img = GrayImage::from_vec(shape[1] as u32, shape[0] as u32, bg_img.to_vec())
            .expect("fail to cast input bg_img to GrayImage");

        let res = self.random_change_bgcolor(&bg_img);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([shape[0], shape[1]]).unwrap();

        reshape_py
    }

    #[pyo3(name = "poisson_edit")]
    pub fn poisson_edit_py<'py>(
        &self,
        font_img: PyReadonlyArray2<'py, u8>,
        bg_img: PyReadonlyArray2<'py, u8>,
        _py: Python<'py>,
    ) -> &'py PyArray2<u8> {
        let shape_font = font_img.shape();
        let shape_bg = bg_img.shape();
        let font_img = font_img.as_slice().expect("fail to read input `font_img`");
        let font_img = GrayImage::from_vec(
            shape_font[1] as u32,
            shape_font[0] as u32,
            font_img.to_vec(),
        )
        .expect("fail to cast input font_img to GrayImage");
        let bg_img = bg_img.as_slice().expect("fail to read input `bg_img`");
        let bg_img = GrayImage::from_vec(shape_bg[1] as u32, shape_bg[0] as u32, bg_img.to_vec())
            .expect("fail to cast input bg_img to GrayImage");

        let res = self.poisson_edit(&font_img, &bg_img);

        let res_py = PyArray::from_vec(_py, res.into_vec());
        let reshape_py = res_py.reshape([shape_bg[0], shape_bg[1]]).unwrap();

        reshape_py
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_change_bg_color() {
        let img = image::open("synth_text/background/3.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let merge_util = MergeUtil {
            height_diff: Random::new_gaussian(2.0, 10.0),
            bg_alpha: Random::new_gaussian(0.5, 1.5),
            bg_beta: Random::new_gaussian(-50.0, 50.0),
            font_alpha: Random::new_uniform(0.2, 1.0),
            reverse_prob: 0.5,
        };

        let start = Instant::now();
        let res = merge_util.random_change_bgcolor(&gray);
        println!("change bg color elapsed: {}", start.elapsed().as_secs_f64());

        res.save("./test-img/bg_color.png").unwrap();
    }

    #[test]
    fn test_random_pad() {
        let img = image::open("./test-img/warp.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let merge_util = MergeUtil {
            height_diff: Random::new_gaussian(2.0, 10.0),
            bg_alpha: Random::new_gaussian(0.5, 1.5),
            bg_beta: Random::new_gaussian(-50.0, 50.0),
            font_alpha: Random::new_uniform(0.2, 1.0),
            reverse_prob: 0.5,
        };

        let start = Instant::now();
        let res = merge_util.random_pad(&gray, 64, 1000);
        println!("random pad elapsed: {}", start.elapsed().as_secs_f64());

        res.save("./test-img/random_pad.png").unwrap();
    }

    #[test]
    fn test_poisson_editing() {
        let img = image::open("./test-img/box.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let merge_util = MergeUtil {
            height_diff: Random::new_gaussian(2.0, 10.0),
            bg_alpha: Random::new_gaussian(0.5, 1.5),
            bg_beta: Random::new_gaussian(-50.0, 50.0),
            font_alpha: Random::new_uniform(0.2, 1.0),
            reverse_prob: 0.5,
        };
        let bg_factory = BgFactory::new("synth_text/background", 64, 1000);

        let start = Instant::now();
        let res = merge_util.poisson_edit(&gray, bg_factory.random());
        println!("random pad elapsed: {}", start.elapsed().as_secs_f64());

        res.save("./test-img/poisson_editing.png").unwrap();
    }

    #[test]
    fn test_background() {
        let bg_factory = BgFactory::new("synth_text/background", 64, 1000);
        let start = Instant::now();
        let a = &bg_factory[7];
        println!(
            "background factory elapsed: {}",
            start.elapsed().as_secs_f64()
        );
        a.save("./test-img/tmp1.png").unwrap();
    }

    #[test]
    fn test_background_random() {
        let bg_factory = BgFactory::new("synth_text/background", 64, 1000);
        let start = Instant::now();
        let a = bg_factory.random();
        println!(
            "background factory elapsed: {}",
            start.elapsed().as_secs_f64()
        );
        a.save("./test-img/tmp1.png").unwrap();
    }
}
