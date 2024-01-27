use indexmap::IndexMap;
use rand_distr::WeightedAliasIndex;

use crate::{font_util::FontUtil, utils::InternalAttrsOwned};

pub fn init_ch_dict<'a, 'b, I: Iterator<Item = &'b S>, S: AsRef<str> + 'b + ?Sized>(
    font_util: &mut FontUtil,
    full_font_list: &'a Vec<InternalAttrsOwned>,
    ch_list: I,
) -> IndexMap<&'b str, Vec<InternalAttrsOwned>> {
    let mut ch_list: Vec<_> = ch_list.map(|ch_str| (ch_str, vec![])).collect();

    for (ch_str, ch_font_list) in ch_list.iter_mut() {
        for font_attrs in full_font_list.iter() {
            if ch_str
                .as_ref()
                .chars()
                .all(|each_ch| font_util.is_font_contain_ch(font_attrs.as_attrs(), each_ch))
                && !ch_font_list.contains(font_attrs)
            {
                ch_font_list.push(font_attrs.clone());
            }
        }
    }

    let ch_list: IndexMap<&str, Vec<InternalAttrsOwned>> = ch_list
        .into_iter()
        .map(|(ch, font_list)| (ch.as_ref(), font_list))
        .collect();

    ch_list
}

enum Frequence {
    NUM(f64),
    MIN,
}

pub fn init_ch_dict_and_weight<'a, 'b>(
    font_util: &mut FontUtil,
    full_font_list: &'a Vec<InternalAttrsOwned>,
    character_file_data: &'b str,
) -> (
    IndexMap<&'b str, Vec<InternalAttrsOwned>>,
    WeightedAliasIndex<f64>,
) {
    let mut is_all_freq_empty = true;
    let mut ch_list_and_weight: Vec<_> = character_file_data
        .trim()
        .split("\n")
        .map(|each| {
            let mut split = each.trim().split("\t");
            let first = split.next().unwrap();
            let second = match split.next() {
                Some(value) => {
                    is_all_freq_empty = false;
                    let value = value.parse::<f64>().unwrap();
                    if value <= 0.0 {
                        Frequence::MIN
                    } else {
                        Frequence::NUM(value)
                    }
                }
                None => Frequence::MIN,
            };

            (first, second, vec![])
        })
        .collect();

    for (ch_str, _, ch_font_list) in ch_list_and_weight.iter_mut() {
        for font_attrs in full_font_list.iter() {
            if ch_str
                .chars()
                .all(|each_ch| font_util.is_font_contain_ch(font_attrs.as_attrs(), each_ch))
                && !ch_font_list.contains(font_attrs)
            {
                ch_font_list.push(font_attrs.clone());
            }
        }
    }

    let ch_list_weights = WeightedAliasIndex::new(
        ch_list_and_weight
            .iter()
            .map(|(_, weight, _)| match weight {
                Frequence::NUM(value) => *value,
                Frequence::MIN => {
                    if is_all_freq_empty {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let ch_list: IndexMap<&str, Vec<InternalAttrsOwned>> = ch_list_and_weight
        .into_iter()
        .map(|(ch, _, font_list)| (ch, font_list))
        .collect();

    (ch_list, ch_list_weights)
}
