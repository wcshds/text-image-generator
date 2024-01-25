use std::{fs, path::Path};

use pyo3::pyclass;
use serde::{Deserialize, Serialize};

use super::effect_helper::math::Random;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Config {
    // 1. font_util
    pub font_dir: String,
    pub chinese_ch_file_path: String,
    pub main_font_list_file_path: String,
    pub latin_corpus_file_path: String,
    pub symbol_file_path: String,
    pub font_size: usize,
    pub line_height: usize,
    pub font_img_height: usize,
    pub font_img_width: usize,
    // 2. cv_util
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
    // 3. merge_util
    pub bg_dir: String,
    pub bg_height: usize,
    pub bg_width: usize,
    pub height_diff: Random,
    pub bg_alpha: Random,
    pub bg_beta: Random,
    pub font_alpha: Random,
    pub reverse_prob: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            font_dir: "./font".to_string(),
            chinese_ch_file_path: "./ch.txt".to_string(),
            main_font_list_file_path: "./symbol.txt".to_string(),
            latin_corpus_file_path: "".to_string(),
            symbol_file_path: "".to_string(),
            font_size: 50,
            line_height: 64,
            font_img_width: 2000,
            font_img_height: 64,
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
            bg_dir: "./synth_text/background".to_string(),
            bg_height: 64,
            bg_width: 1000,
            height_diff: Random::new_uniform(2.0, 10.0),
            bg_alpha: Random::new_gaussian(0.5, 1.5),
            bg_beta: Random::new_gaussian(-50.0, 50.0),
            font_alpha: Random::new_uniform(0.2, 1.0),
            reverse_prob: 0.5,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct FontYaml {
    font_dir: String,
    chinese_ch_file_path: String,
    main_font_list_file_path: String,
    #[serde(default)]
    latin_corpus_file_path: String,
    #[serde(default)]
    symbol_file_path: String,
    font_size: usize,
    line_height: usize,
    font_img_height: usize,
    font_img_width: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct RandomYaml(f64, f64, String);

impl RandomYaml {
    fn to_random(&self) -> Random {
        if self.2 == "g" {
            Random::new_gaussian(self.0, self.1)
        } else if self.2 == "u" {
            Random::new_uniform(self.0, self.1)
        } else {
            panic!("distribution parameter in config file should be `g` or `u`");
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct CvYaml {
    box_prob: f64,
    perspective_prob: f64,
    perspective_x: RandomYaml,
    perspective_y: RandomYaml,
    perspective_z: RandomYaml,
    blur_prob: f64,
    blur_sigma: RandomYaml,
    filter_prob: f64,
    emboss_prob: f64,
    sharp_prob: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct MergeYaml {
    pub bg_dir: String,
    pub bg_height: usize,
    pub bg_width: usize,
    // make it into Random(2.0, height_diff) later
    pub height_diff: f64,
    pub bg_alpha: RandomYaml,
    pub bg_beta: RandomYaml,
    pub font_alpha: RandomYaml,
    pub reverse_prob: f64,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "UPPERCASE")]
struct GeneratorConfigYaml {
    font: FontYaml,
    cv: CvYaml,
    merge: MergeYaml,
}

impl Config {
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Config {
        let yaml_str = fs::read_to_string(path).expect("the config file does not exist");
        let yaml: GeneratorConfigYaml =
            serde_yaml::from_str(&yaml_str).expect("fail to parse config file");

        Config {
            font_dir: yaml.font.font_dir,
            chinese_ch_file_path: yaml.font.chinese_ch_file_path,
            main_font_list_file_path: yaml.font.main_font_list_file_path,
            latin_corpus_file_path: yaml.font.latin_corpus_file_path,
            symbol_file_path: yaml.font.symbol_file_path,
            font_size: yaml.font.font_size,
            line_height: yaml.font.line_height,
            font_img_width: yaml.font.font_img_width,
            font_img_height: yaml.font.font_img_height,
            box_prob: yaml.cv.box_prob,
            perspective_prob: yaml.cv.perspective_prob,
            perspective_x: yaml.cv.perspective_x.to_random(),
            perspective_y: yaml.cv.perspective_y.to_random(),
            perspective_z: yaml.cv.perspective_z.to_random(),
            blur_prob: yaml.cv.blur_prob,
            blur_sigma: yaml.cv.blur_sigma.to_random(),
            filter_prob: yaml.cv.filter_prob,
            emboss_prob: yaml.cv.emboss_prob,
            sharp_prob: yaml.cv.sharp_prob,
            bg_dir: yaml.merge.bg_dir,
            bg_height: yaml.merge.bg_height,
            bg_width: yaml.merge.bg_width,
            height_diff: Random::new_uniform(2.0, yaml.merge.height_diff),
            bg_alpha: yaml.merge.bg_alpha.to_random(),
            bg_beta: yaml.merge.bg_beta.to_random(),
            font_alpha: yaml.merge.font_alpha.to_random(),
            reverse_prob: yaml.merge.reverse_prob,
        }
    }
}
