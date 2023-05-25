use std::fs;

use corpus::{get_random_chinese_text_with_font_list, wrap_text_with_font_list};
use cosmic_text::{
    Attrs, AttrsList, Buffer, BufferLine, Color, Family, FontSystem, Metrics, Style, SwashCache,
    Weight,
};
use font_util::FontUtil;
use image_process::generate_image;
use indexmap::IndexMap;
use numpy::{PyArray, PyArray3};
use pyo3::{prelude::*, types::PyList};
use rand_distr::WeightedAliasIndex;
use utils::IndexMapStrUtils;

use crate::{
    init::{init_ch_dict, init_ch_dict_and_weight},
    utils::StringUsefulUtils,
};

pub mod corpus;
pub mod font_util;
pub mod image_process;
pub mod init;
pub mod utils;

#[pyclass]
struct Generator {
    font_system: FontSystem,
    font_util: FontUtil,
    editor_buffer: Buffer,
    swash_cache: SwashCache,
    #[pyo3(get)]
    font_list: Vec<String>,
    #[pyo3(get)]
    chinese_ch_dict: IndexMap<String, Vec<String>>,
    chinese_ch_weights: WeightedAliasIndex<f64>,
    #[pyo3(get)]
    latin_corpus: Option<String>,
    symbol: Option<Vec<String>>,
    #[pyo3(get)]
    latin_ch_dict: Option<IndexMap<String, Vec<String>>>,
    #[pyo3(get)]
    symbol_dict: Option<IndexMap<String, Vec<String>>>,
    #[pyo3(get)]
    main_font_list: Vec<String>, // 若字符的字體列表爲空，則隨機從 main_font_list 中擇一字體
}

#[pymethods]
impl Generator {
    #[new]
    #[pyo3(signature = (font_dir="./font", main_font_list_file=None, chinese_ch_file="./chinese_ch.txt", latin_corpus_file=None, symbol_file=None))]
    fn py_new(
        font_dir: &str,
        main_font_list_file: Option<&str>,
        chinese_ch_file: &str,
        latin_corpus_file: Option<&str>,
        symbol_file: Option<&str>,
    ) -> PyResult<Self> {
        let mut font_system = FontSystem::new();
        let db = font_system.db_mut();
        db.load_fonts_dir(font_dir);

        // 加載 latin 語料文件
        let latin_corpus_file_data = if let Some(latin_corpus_file) = latin_corpus_file {
            let data = fs::read_to_string(latin_corpus_file).unwrap();
            Some(data)
        } else {
            None
        };

        // 加載 symbol 文件
        let symbol_file_data = if let Some(symbol_file) = symbol_file {
            let data: Vec<_> = fs::read_to_string(symbol_file)
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
            chinesecharacter_file_data = fs::read_to_string(chinese_ch_file).unwrap();
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

        let mut buffer = Buffer::new(&mut font_system, Metrics::new(50.0, 64.0));
        buffer.set_size(&mut font_system, 2500.0, 64.0);

        let chinese_ch_dict = chinese_ch_dict.clone_to_string();
        let latin_ch_dict = if let Some(content) = latin_ch_dict {
            Some(content.clone_to_string())
        } else {
            None
        };
        let symbol_dict = if let Some(content) = symbol_dict {
            Some(content.clone_to_string())
        } else {
            None
        };

        let main_font_list: Vec<_> = if let Some(main_font_list_file) = main_font_list_file {
            fs::read_to_string(main_font_list_file)
                .unwrap()
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
            chinese_ch_dict,
            chinese_ch_weights,
            latin_corpus: latin_corpus_file_data,
            symbol: symbol_file_data,
            latin_ch_dict,
            symbol_dict,
            main_font_list,
        })
    }

    fn set_latin_ch_dict(&mut self, ch: String, font_list: Vec<String>) {
        if let Some(content) = &mut self.latin_ch_dict {
            *content.entry(ch).or_insert(vec![]) = font_list;
        }
    }

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
                    list.as_ref(py).append((ch, content)).unwrap();
                } else {
                    list.as_ref(py)
                        .append::<(&str, &Vec<String>)>((ch, &vec![]))
                        .unwrap();
                }
            }

            Ok(list)
        })
    }

    // min: 指定生成文本的字數下限
    // max: 指定生成文本的字數上限
    // add_extra_symbol: 是否額外爲生成文本增加標點
    fn wrap_text_with_font_list(&self, text: &str) -> PyResult<Py<PyList>> {
        let chinese_text_with_font_list = wrap_text_with_font_list(text, &self.chinese_ch_dict);
        Python::with_gil(|py| -> PyResult<Py<PyList>> {
            let list: Py<PyList> = PyList::empty(py).into();
            for (ch, font_list) in chinese_text_with_font_list {
                if let Some(content) = font_list {
                    list.as_ref(py).append((ch, content)).unwrap();
                } else {
                    list.as_ref(py)
                        .append::<(&str, &Vec<String>)>((ch, &vec![]))
                        .unwrap();
                }
            }

            Ok(list)
        })
    }

    #[pyo3(signature = (text_with_font_list, text_color=(0, 0, 0), background_color=(255, 255, 255)))]
    fn gen_image_from_text_with_font_list(
        &mut self,
        text_with_font_list: Vec<(String, Vec<String>)>,
        text_color: (u8, u8, u8),
        background_color: (u8, u8, u8),
    ) -> PyResult<Py<PyArray3<u8>>> {
        self.editor_buffer.lines.clear();

        let attrs = Attrs::new()
            .family(Family::Name("Gandhari Unicode"))
            .style(Style::Normal)
            .weight(Weight::NORMAL);

        let temp: Vec<_> = text_with_font_list
            .iter()
            .map(|(ch, font_list)| (ch, Some(font_list)))
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

        self.editor_buffer
            .lines
            .push(BufferLine::new(&line_text, attrs_list));

        self.editor_buffer.shape_until_scroll(&mut self.font_system);

        let text_color = Color::rgb(text_color.0, text_color.1, text_color.2);
        let background_color =
            image::Rgb([background_color.0, background_color.1, background_color.2]);

        let img = generate_image(
            &mut self.editor_buffer,
            &mut self.font_system,
            &mut self.swash_cache,
            text_color,
            background_color,
        );

        // img.save("internal.png").unwrap();

        let img_height = img.height() as usize;
        let img_width = img.width() as usize;

        let raw = img.into_vec();

        Python::with_gil(|py| -> PyResult<Py<PyArray3<u8>>> {
            let initial = PyArray::from_vec(py, raw);
            let res: Py<PyArray3<u8>> = initial.reshape([img_height, img_width, 3]).unwrap().into();

            Ok(res)
        })
    }
}

#[pymodule]
fn text_image_generator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Generator>()?;
    Ok(())
}
