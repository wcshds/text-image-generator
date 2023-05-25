use cosmic_text::{Attrs, Family};
use indexmap::IndexMap;
use rand_distr::WeightedAliasIndex;

use crate::font_util::FontUtil;

pub fn init_ch_dict<'a, 'b, I: Iterator<Item = &'b S>, S: AsRef<str> + 'b + ?Sized>(
    font_util: &mut FontUtil,
    full_font_list: &'a Vec<String>,
    ch_list: I,
) -> IndexMap<&'b str, Vec<&'a str>> {
    let mut ch_list: Vec<_> = ch_list.map(|ch_str| (ch_str, vec![])).collect();

    for (ch_str, ch_font_list) in ch_list.iter_mut() {
        for font_name in full_font_list.iter() {
            if ch_str.as_ref().chars().all(|each_ch| {
                font_util.is_font_contain_ch(Attrs::new().family(Family::Name(font_name)), each_ch)
            }) && !ch_font_list.contains(&&font_name[..])
            {
                ch_font_list.push(&font_name[..]);
            }
        }
    }

    let ch_list: IndexMap<&str, Vec<&str>> = ch_list
        .into_iter()
        .map(|(ch, font_list)| (ch.as_ref(), font_list))
        .collect();

    ch_list
}

pub fn init_ch_dict_and_weight<'a, 'b>(
    font_util: &mut FontUtil,
    full_font_list: &'a Vec<String>,
    character_file_data: &'b str,
) -> (IndexMap<&'b str, Vec<&'a str>>, WeightedAliasIndex<f64>) {
    let mut ch_list_and_weight: Vec<_> = character_file_data
        .trim()
        .split("\n")
        .map(|each| {
            let mut split = each.trim().split("\t");
            let first = split.next().unwrap();
            let second = split.next().unwrap().parse::<f64>().unwrap();
            (first, second, vec![])
        })
        .collect();

    for (ch_str, _, ch_font_list) in ch_list_and_weight.iter_mut() {
        for font_name in full_font_list.iter() {
            if ch_str.chars().all(|each_ch| {
                font_util.is_font_contain_ch(Attrs::new().family(Family::Name(font_name)), each_ch)
            }) && !ch_font_list.contains(&&font_name[..])
            {
                ch_font_list.push(&&font_name[..]);
            }
        }
    }

    let ch_list_weights = WeightedAliasIndex::new(
        ch_list_and_weight
            .iter()
            .map(|(_, weight, _)| weight.clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let ch_list: IndexMap<&str, Vec<&str>> = ch_list_and_weight
        .into_iter()
        .map(|(ch, _, font_list)| (ch, font_list))
        .collect();

    (ch_list, ch_list_weights)
}
