use std::{collections::HashMap, str::from_utf8_unchecked};

use cosmic_text::{Attrs, AttrsOwned, Family, Stretch, Style, Weight};
use indexmap::IndexMap;
use pyo3::{IntoPy, PyObject, Python};

pub trait StringUsefulUtils {
    fn dedup(&self) -> String;
    fn dedup_to_vec(&self) -> Vec<&str>;
}

impl<S: AsRef<str>> StringUsefulUtils for S {
    fn dedup_to_vec(&self) -> Vec<&str> {
        let mut reserve: HashMap<&str, i32> = HashMap::new();
        let bytes = self.as_ref().as_bytes();
        let total_len = bytes.len();
        let mut idx = 0;
        while idx < total_len {
            let byte = bytes[idx];
            if !utf8_width::is_width_0(byte) {
                let ch_len = unsafe { utf8_width::get_width_assume_valid(byte) };
                let ch = unsafe { from_utf8_unchecked(&bytes[idx..idx + ch_len]) };
                reserve.entry(ch).or_default();

                idx += ch_len;
            } else {
                idx += 1;
            }
        }

        let mut res = reserve.keys().copied().collect::<Vec<_>>();
        res.sort();
        res
    }

    fn dedup(&self) -> String {
        let dedup_vec = self.dedup_to_vec();
        let res_len = dedup_vec.len();
        let mut res = String::with_capacity(res_len * 3);
        for each in dedup_vec {
            res.push_str(each)
        }

        res
    }
}

pub trait IndexMapStrUtils {
    fn as_tuple_infos(&self) -> IndexMap<String, Vec<(String, u16, u16, u16)>>;
}

impl<S1> IndexMapStrUtils for IndexMap<S1, &Vec<AttrsOwned>>
where
    S1: AsRef<str>,
{
    fn as_tuple_infos(&self) -> IndexMap<String, Vec<(String, u16, u16, u16)>> {
        self.iter()
            .map(|(ch, font_list)| {
                (
                    ch.as_ref().to_string(),
                    font_list
                        .iter()
                        .map(|font_attrs_owned| attrs_owned_to_tuple(font_attrs_owned))
                        .collect(),
                )
            })
            .collect()
    }
}

impl<S1> IndexMapStrUtils for IndexMap<S1, Vec<AttrsOwned>>
where
    S1: AsRef<str>,
{
    fn as_tuple_infos(&self) -> IndexMap<String, Vec<(String, u16, u16, u16)>> {
        self.iter()
            .map(|(ch, font_list)| {
                (
                    ch.as_ref().to_string(),
                    font_list
                        .iter()
                        .map(|font_attrs_owned| attrs_owned_to_tuple(font_attrs_owned))
                        .collect(),
                )
            })
            .collect()
    }
}

pub fn attrs_owned_to_tuple(attrs_owned: &AttrsOwned) -> (String, u16, u16, u16) {
    let attrs = attrs_owned.as_attrs();
    let font_name = match attrs.family {
        Family::Name(name) => name.to_string(),
        Family::Serif => "FamilySerif".to_string(),
        Family::SansSerif => "FamilySansSerif".to_string(),
        Family::Cursive => "FamilyCursive".to_string(),
        Family::Fantasy => "FamilyFantasy".to_string(),
        Family::Monospace => "FamilyMonospace".to_string(),
    };
    let font_style: u16 = match attrs.style {
        Style::Normal => 0,
        Style::Italic => 1,
        Style::Oblique => 2,
    };
    let font_weight = attrs.weight.0;
    let font_stretch = attrs.stretch.to_number();

    (font_name, font_style, font_weight, font_stretch)
}

#[derive(Clone, Debug)]
pub struct InternalAttrsOwned {
    attrs_owned: AttrsOwned,
}

impl InternalAttrsOwned {
    pub fn new(attrs_owned: AttrsOwned) -> Self {
        Self { attrs_owned }
    }

    pub fn to_tuple(&self) -> (String, u16, u16, u16) {
        attrs_owned_to_tuple(&self.attrs_owned)
    }

    pub fn from_tuple(src: (String, u16, u16, u16)) -> Self {
        let family = match &src.0[..] {
            "FamilySerif" => Family::Serif,
            "FamilySansSerif" => Family::SansSerif,
            "FamilyCursive" => Family::Cursive,
            "FamilyFantasy" => Family::Fantasy,
            "FamilyMonospace" => Family::Monospace,
            _ => Family::Name(&src.0[..]),
        };
        let style = match src.1 {
            0 => Style::Normal,
            1 => Style::Italic,
            2 => Style::Oblique,
            _ => panic!("font style should be 1 to 3"),
        };
        let weight = Weight(src.2);
        let stretch = match src.3 {
            1 => Stretch::UltraCondensed,
            2 => Stretch::ExtraCondensed,
            3 => Stretch::Condensed,
            4 => Stretch::SemiCondensed,
            5 => Stretch::Normal,
            6 => Stretch::SemiExpanded,
            7 => Stretch::Expanded,
            8 => Stretch::ExtraExpanded,
            9 => Stretch::UltraExpanded,
            _ => panic!("font stretch should be 1 to 9"),
        };

        let attrs = Attrs::new()
            .family(family)
            .weight(weight)
            .stretch(stretch)
            .style(style);

        Self {
            attrs_owned: AttrsOwned::new(attrs),
        }
    }

    pub fn as_attrs(&self) -> Attrs {
        self.attrs_owned.as_attrs()
    }
}

impl PartialEq for InternalAttrsOwned {
    fn eq(&self, other: &Self) -> bool {
        self.attrs_owned == other.attrs_owned
    }
}

impl IntoPy<PyObject> for InternalAttrsOwned {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let res = self.to_tuple();
        res.into_py(py)
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use super::*;

    #[test]
    fn test_string_dedup() {
        let data = fs::read_to_string("./latin_corpus.txt").unwrap();
        let result = data.dedup();

        println!("{:#?}", result);
    }
}
