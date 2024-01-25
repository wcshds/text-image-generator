use std::{ops::RangeInclusive, str::from_utf8_unchecked};

use indexmap::IndexMap;
use rand::{self, seq::SliceRandom, Rng};
use rand_distr::{Distribution, WeightedAliasIndex};

use crate::utils::InternalAttrsOwned;

pub fn get_random_french_text<'a, S1, S2, S3>(
    ch_dict: &'a IndexMap<S1, Vec<S2>>,
    weights: &WeightedAliasIndex<f64>,
    symbol: Option<&'a Vec<S3>>,
    range: RangeInclusive<u32>,
) -> Vec<(&'a str, Option<&'a Vec<S2>>)>
where
    S1: AsRef<str>,
    S2: AsRef<str>,
    S3: AsRef<str>,
{
    let mut rng = rand::thread_rng();

    let num = rng.gen_range(range);

    let mut res = Vec::with_capacity(150);
    if let Some(symbol_content) = symbol {
        let insert_idx = rng.gen_range(2..=num);
        let symbol = symbol_content.choose(&mut rng).unwrap();
        for i in 1..=num {
            if i == insert_idx {
                res.push((symbol.as_ref(), None));
            }

            let (temp_ch, temp_font_list) = ch_dict.get_index(weights.sample(&mut rng)).unwrap();
            res.push((temp_ch.as_ref(), Some(temp_font_list)));
        }
    } else {
        for _ in 1..=num {
            let (temp_ch, temp_font_list) = ch_dict.get_index(weights.sample(&mut rng)).unwrap();
            res.push((temp_ch.as_ref(), Some(temp_font_list)));
        }
    }

    res
}

pub fn get_random_chinese_text_with_font_list<'a, S1, S2>(
    ch_dict: &'a IndexMap<S1, Vec<InternalAttrsOwned>>,
    weights: &WeightedAliasIndex<f64>,
    symbol: Option<&'a Vec<S2>>,
    range: RangeInclusive<u32>,
) -> Vec<(&'a str, Option<&'a Vec<InternalAttrsOwned>>)>
where
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    let mut rng = rand::thread_rng();

    let num = rng.gen_range(range);

    let mut res = Vec::with_capacity(15);
    if let Some(symbol_content) = symbol {
        let insert_idx = rng.gen_range(2..=num);
        let symbol = symbol_content.choose(&mut rng).unwrap();
        for i in 1..=num {
            if i == insert_idx {
                res.push((symbol.as_ref(), None));
            }

            let (temp_ch, temp_font_list) = ch_dict.get_index(weights.sample(&mut rng)).unwrap();
            res.push((temp_ch.as_ref(), Some(temp_font_list)));
        }
    } else {
        for _ in 1..=num {
            let (temp_ch, temp_font_list) = ch_dict.get_index(weights.sample(&mut rng)).unwrap();
            res.push((temp_ch.as_ref(), Some(temp_font_list)));
        }
    }

    res
}

pub fn wrap_text_with_font_list<'a, 'b, S1, S2>(
    text: &'a S1,
    ch_dict: &'b IndexMap<S2, Vec<InternalAttrsOwned>>,
) -> Vec<(&'a str, Option<&'b Vec<InternalAttrsOwned>>)>
where
    S1: AsRef<str> + ?Sized,
    S2: std::hash::Hash + std::cmp::Eq + std::borrow::Borrow<str>,
{
    let bytes = text.as_ref().as_bytes();
    let mut res = vec![];

    let length = bytes.len();
    let mut idx = 0;
    while idx < length {
        if !utf8_width::is_width_0(bytes[idx]) {
            let ch_bytes_length = unsafe { utf8_width::get_width_assume_valid(bytes[idx]) };
            let ch = unsafe { from_utf8_unchecked(&bytes[idx..idx + ch_bytes_length]) };
            res.push((ch, ch_dict.get(ch)));
            idx += ch_bytes_length;
        } else {
            idx += 1;
        }
    }

    res
}

#[cfg(test)]
mod test {
    use std::fs;

    use cosmic_text::FontSystem;

    use crate::{font_util::FontUtil, init::init_ch_dict_and_weight};

    use super::*;

    #[test]
    fn test_wrap_text_with_font_list() {
        let mut font_system = FontSystem::new();
        let db = font_system.db_mut();
        db.load_fonts_dir("./font");
        let mut fu = FontUtil::new(&font_system);
        let full_font_list = fu.get_full_font_list();
        let character_file_data = fs::read_to_string("./ch.txt").unwrap();
        let (ch_dict, _) = init_ch_dict_and_weight(&mut fu, &full_font_list, &character_file_data);

        println!("{:?}", wrap_text_with_font_list("這是一個測試", &ch_dict));
    }
}
