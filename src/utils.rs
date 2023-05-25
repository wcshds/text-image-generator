use std::{collections::HashMap, str::from_utf8_unchecked};

use indexmap::IndexMap;

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
    fn clone_to_string(&self) -> IndexMap<String, Vec<String>>;
}

impl<S1, S2> IndexMapStrUtils for IndexMap<S1, Vec<S2>>
where
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    fn clone_to_string(&self) -> IndexMap<String, Vec<String>> {
        self.iter()
            .map(|(ch, font_list)| {
                (
                    ch.as_ref().to_string(),
                    font_list
                        .iter()
                        .map(|font| font.as_ref().to_string())
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
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
