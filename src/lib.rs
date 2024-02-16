use std::fs;

use corpus::{get_random_chinese_text_with_font_list, wrap_text_with_font_list};
use cosmic_text::{
    Attrs, AttrsList, Buffer, BufferLine, Color, Family, FontSystem, Metrics, Style, SwashCache,
    Weight,
};
use cv_util::CvUtil;
use font_util::FontUtil;
use image_process::generate_image;
use indexmap::IndexMap;
use merge_util::{BgFactory, MergeUtil};
use numpy::{PyArray, PyArrayDyn};
use parse_config::Config;
use pyo3::{prelude::*, types::PyList};
use rand_distr::WeightedAliasIndex;
use utils::InternalAttrsOwned;

use crate::{
    init::{init_ch_dict, init_ch_dict_and_weight},
    utils::StringUsefulUtils,
};

pub mod corpus;
pub mod cv_util;
pub mod effect_helper;
pub mod font_util;
pub mod image_process;
pub mod init;
pub mod merge_util;
pub mod parse_config;
pub mod utils;

#[pyclass]
struct Generator {
    font_system: FontSystem,
    font_util: FontUtil,
    editor_buffer: Buffer,
    swash_cache: SwashCache,
    #[pyo3(get)]
    cv_util: CvUtil,
    #[pyo3(get)]
    merge_util: MergeUtil,
    #[pyo3(get)]
    bg_factory: BgFactory,
    #[pyo3(get)]
    font_list: Vec<InternalAttrsOwned>,
    #[pyo3(get)]
    chinese_ch_dict: IndexMap<String, Vec<InternalAttrsOwned>>,
    chinese_ch_weights: WeightedAliasIndex<f64>,
    #[pyo3(get)]
    latin_corpus: Option<String>,
    symbol: Option<Vec<String>>,
    #[pyo3(get)]
    latin_ch_dict: Option<IndexMap<String, Vec<InternalAttrsOwned>>>,
    #[pyo3(get)]
    symbol_dict: Option<IndexMap<String, Vec<InternalAttrsOwned>>>,
    #[pyo3(get)]
    main_font_list: Vec<String>, // 若字符的字體列表爲空，則隨機從 main_font_list 中擇一字體
}

#[pymethods]
impl Generator {
    #[new]
    #[pyo3(signature = (config_path="./config.yaml"))]
    fn py_new(config_path: &str) -> PyResult<Self> {
        let config = Config::from_yaml(config_path);

        let mut font_system = FontSystem::new();
        let db = font_system.db_mut();
        db.load_fonts_dir(&config.font_dir);

        // 加載 latin 語料文件
        let latin_corpus_file_data = if config.latin_corpus_file_path.len() > 0 {
            let data = fs::read_to_string(&config.latin_corpus_file_path).unwrap();
            Some(data)
        } else {
            None
        };

        // 加載 symbol 文件
        let symbol_file_data = if config.symbol_file_path.len() > 0 {
            let data: Vec<_> = fs::read_to_string(&config.symbol_file_path)
                .unwrap()
                .trim_matches('\n')
                .split("\n")
                .map(String::from)
                .collect();
            Some(data)
        } else {
            None
        };

        let (
            full_font_list,
            chinesecharacter_file_data,
            chinese_ch_dict,
            chinese_ch_weights,
            latin_ch_dict,
            symbol_dict,
        );

        {
            let mut font_util = font_util::FontUtil::new(&font_system);
            full_font_list = font_util.get_full_font_list();
            chinesecharacter_file_data = fs::read_to_string(config.chinese_ch_file_path).unwrap();
            println!("正在分析字體所包含的字符...");
            (chinese_ch_dict, chinese_ch_weights) = init_ch_dict_and_weight(
                &mut font_util,
                &full_font_list,
                &chinesecharacter_file_data,
            );

            latin_ch_dict = if let Some(ref latin_corpus_file_data) = latin_corpus_file_data {
                let temp = latin_corpus_file_data.dedup_to_vec().into_iter();
                Some(init_ch_dict(&mut font_util, &full_font_list, temp))
            } else {
                None
            };

            symbol_dict = if let Some(ref symbol_file_data) = symbol_file_data {
                let data = symbol_file_data.iter();
                Some(init_ch_dict(&mut font_util, &full_font_list, data))
            } else {
                None
            };

            println!("分析完成!");
        }

        let font_util = font_util::FontUtil::new(&font_system);

        // create one per application
        let swash_cache = SwashCache::new();

        let mut buffer = Buffer::new(
            &mut font_system,
            Metrics::new(config.font_size as f32, config.line_height as f32),
        );
        buffer.set_size(
            &mut font_system,
            config.font_img_width as f32,
            config.font_img_height as f32,
        );

        let main_font_list: Vec<_> = if config.main_font_list_file_path.len() > 0 {
            fs::read_to_string(&config.main_font_list_file_path)
                .unwrap()
                .trim()
                .split("\n")
                .map(String::from)
                .collect()
        } else {
            vec![]
        };

        Ok(Self {
            font_system,
            font_util,
            editor_buffer: buffer,
            swash_cache,
            font_list: full_font_list,
            chinese_ch_dict: chinese_ch_dict
                .into_iter()
                .map(|(ch, dic)| (ch.to_string(), dic))
                .collect(),
            chinese_ch_weights,
            latin_corpus: latin_corpus_file_data.clone(),
            symbol: symbol_file_data.clone(),
            latin_ch_dict: if let Some(ch_dict) = latin_ch_dict {
                Some(
                    ch_dict
                        .into_iter()
                        .map(|(ch, dic)| (ch.to_string(), dic.clone()))
                        .collect(),
                )
            } else {
                None
            },
            symbol_dict: if let Some(symbol_dict) = symbol_dict {
                Some(
                    symbol_dict
                        .into_iter()
                        .map(|(ch, dic)| (ch.to_string(), dic.clone()))
                        .collect(),
                )
            } else {
                None
            },
            main_font_list,
            cv_util: CvUtil {
                box_prob: config.box_prob,
                perspective_prob: config.perspective_prob,
                perspective_x: config.perspective_x,
                perspective_y: config.perspective_y,
                perspective_z: config.perspective_z,
                blur_prob: config.blur_prob,
                blur_sigma: config.blur_sigma,
                filter_prob: config.filter_prob,
                emboss_prob: config.emboss_prob,
                sharp_prob: config.sharp_prob,
            },
            merge_util: MergeUtil {
                height_diff: config.height_diff,
                bg_alpha: config.bg_alpha,
                bg_beta: config.bg_beta,
                font_alpha: config.font_alpha,
                reverse_prob: config.reverse_prob,
            },
            bg_factory: BgFactory::new(config.bg_dir, config.bg_height, config.bg_width),
        })
    }

    fn set_bg_size(&mut self, height: usize, width: usize) {
        self.bg_factory = BgFactory::new(&self.bg_factory.bg_dir, height, width);
    }

    // fn set_latin_ch_dict(&mut self, ch: String, font_list: Vec<String>) {
    //     if let Some(content) = &mut self.latin_ch_dict {
    //         *content.entry(ch).or_insert(vec![]) = font_list;
    //     }
    // }

    // min: 指定生成文本的字數下限
    // max: 指定生成文本的字數上限
    // add_extra_symbol: 是否額外爲生成文本增加標點
    #[pyo3(signature = (min=5, max=10, add_extra_symbol=false))]
    fn get_random_chinese(
        &self,
        min: u32,
        max: u32,
        add_extra_symbol: bool,
    ) -> PyResult<Py<PyList>> {
        let symbol = if add_extra_symbol {
            self.symbol.as_ref()
        } else {
            None
        };
        let chinese_text_with_font_list = get_random_chinese_text_with_font_list(
            &self.chinese_ch_dict,
            &self.chinese_ch_weights,
            symbol,
            min..=max,
        );
        Python::with_gil(|py| -> PyResult<Py<PyList>> {
            let list: Py<PyList> = PyList::empty(py).into();
            for (ch, font_list) in chinese_text_with_font_list {
                if let Some(content) = font_list {
                    list.as_ref(py)
                        .append((
                            ch,
                            content
                                .iter()
                                .map(|each| each.to_tuple())
                                .collect::<Vec<_>>(),
                        ))
                        .unwrap();
                } else {
                    list.as_ref(py)
                        .append::<(&str, &Vec<String>)>((ch, &vec![]))
                        .unwrap();
                }
            }

            Ok(list)
        })
    }

    fn wrap_text_with_font_list(&self, text: &str) -> PyResult<Py<PyList>> {
        let chinese_text_with_font_list = wrap_text_with_font_list(text, &self.chinese_ch_dict);
        Python::with_gil(|py| -> PyResult<Py<PyList>> {
            let list: Py<PyList> = PyList::empty(py).into();
            for (ch, font_list) in chinese_text_with_font_list {
                if let Some(content) = font_list {
                    list.as_ref(py)
                        .append((
                            ch,
                            content
                                .iter()
                                .map(|each| each.to_tuple())
                                .collect::<Vec<_>>(),
                        ))
                        .unwrap();
                } else {
                    list.as_ref(py)
                        .append::<(&str, &Vec<String>)>((ch, &vec![]))
                        .unwrap();
                }
            }

            Ok(list)
        })
    }

    #[pyo3(signature = (text_with_font_list, text_color=(0, 0, 0), background_color=(255, 255, 255), apply_effect=false))]
    fn gen_image_from_text_with_font_list<'py>(
        &mut self,
        text_with_font_list: Vec<(String, Vec<(String, u16, u16, u16)>)>,
        text_color: (u8, u8, u8),
        background_color: (u8, u8, u8),
        apply_effect: bool,
        _py: Python<'py>,
    ) -> &'py PyArrayDyn<u8> {
        self.editor_buffer.lines.clear();

        let attrs = Attrs::new()
            .family(Family::Name("Gandhari Unicode"))
            .style(Style::Normal)
            .weight(Weight::NORMAL);

        let temp: Vec<_> = text_with_font_list
            .into_iter()
            .map(|(ch, font_list)| {
                (
                    ch,
                    Some(
                        font_list
                            .into_iter()
                            .map(|each| InternalAttrsOwned::from_tuple(each))
                            .collect::<Vec<_>>(),
                    ),
                )
            })
            .collect();
        let temp = temp
            .iter()
            .map(|(ch, font_list)| (ch, font_list.as_ref()))
            .collect();

        let res = self
            .font_util
            .map_chinese_corpus_with_attrs(&temp, &self.main_font_list);

        // let mut line_text = String::with_capacity(text.len());
        let mut line_text = String::new();
        let mut attrs_list = AttrsList::new(attrs);
        for (text, attrs) in res {
            let start = line_text.len();
            line_text.push_str(&text);
            let end = line_text.len();
            attrs_list.add_span(start..end, attrs);
        }

        self.editor_buffer.lines.push(BufferLine::new(
            &line_text,
            attrs_list,
            cosmic_text::Shaping::Advanced,
        ));

        self.editor_buffer
            .shape_until_scroll(&mut self.font_system, false);

        let text_color = Color::rgb(text_color.0, text_color.1, text_color.2);
        let background_color =
            image::Rgb([background_color.0, background_color.1, background_color.2]);

        let (img_width, img_height) = self.editor_buffer.size();
        let img = generate_image(
            &mut self.editor_buffer,
            &mut self.font_system,
            &mut self.swash_cache,
            text_color,
            background_color,
            img_width as usize,
            img_height as usize,
        );

        if apply_effect {
            let gray = image::imageops::grayscale(&img);
            let font_img = self.cv_util.apply_effect(gray);
            let bg_img = self.bg_factory.random();
            let merge_img = self.merge_util.poisson_edit(&font_img, bg_img);

            let img_height = merge_img.height() as usize;
            let img_width = merge_img.width() as usize;

            let raw = merge_img.into_vec();

            let initial = PyArray::from_vec(_py, raw);
            let res = initial.reshape([img_height, img_width]).unwrap();

            return res.to_dyn();
        }

        let img_height = img.height() as usize;
        let img_width = img.width() as usize;

        let raw = img.into_vec();

        let initial = PyArray::from_vec(_py, raw);
        let res = initial.reshape([img_height, img_width, 3]).unwrap();
        res.to_dyn()
    }
}

#[pyclass]
struct ImageEffect {}

#[pymodule]
fn text_image_generator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Generator>()?;
    m.add_class::<BgFactory>()?;
    Ok(())
}
