# 一些碎碎唸

2022 年 7 月 19 日信標委正式發佈了 GB 18030-2022 《信息技術 中文編碼字符集》。該標準將於 2023 年 8 月 1 日開始正式實施。在可以預見的未來，在公共服務中處理生僻字將不再是障礙，對生僻字 OCR 文字識別的需求也將不断增加。

因此，小吳想要針對目前已經編碼的中日韓統一表意文字訓練一個 OCR 模型，方便自己日後使用。由於收集眞實文本圖像作爲訓練集需要人工標註，工作量較大，所以通行的手段是基於多種字體生成文本圖像進行訓練。可惜目前開源的合成文本工具都沒有考慮到對擴展區漢字的支援，且合成图像的速度也不盡如人意，我就索性用 Rust 自己寫一個文本合成工具了。（現在 Rust 的文本整形 crates 都已經很成熟了，讚一個）

這個工具可以和 Pytorch 的 Dataset 組合起來，做到在訓練時實時生成訓練集，避免了訓練集在服務器間傳輸的不便。

# 安裝

這個工具已發布至 PyPI

```
pip install text-image-generator
```

# 編譯

這個工具的編譯需要安裝 Rust，若尚未安裝，請運行：

```
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

然後透過 pip 安裝 maturin 用於構建 pyo3：

```
pip install maturin
```

將本工具的 git 倉庫下載到本地：

```
git clone https://github.com/wcshds/text-image-generator.git
cd text-image-generator
```

生成 wheel：

```
maturin build --release
```

在生成 whl 文件後，透過 pip 命令安裝即可。

```
pip install text_image_generator-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
```

# 使用前的準備

1. 事先需要找到足夠多的字體文件放到目錄中，這些字體文件需要覆蓋想要生成的所有字符。字體主目錄下可以有子目錄，工具會遞歸查找指定的字體主目錄下所有字體文件。**注意：暫時不支援可變字體。**

    字體文件可以在 [這裡](https://drive.google.com/file/d/1dy3ccgcR6SEnfohhZBRoziszsEXav4G2/view?usp=sharing) 下載。

2. 準備一個文本文件，其中包含想要生成的所有字符的全集及其對應的字頻，字符與字頻之間應用製表符分割（字頻可省略），具體格式如下：

```
〇	1
一	1
乙	1
二	1
十	1
丁	1
厂	1
```

3. 準備一個文本文件，其中包含 fallback 字體的名稱，格式如下：

```
SimSun
TW-Sung
KaiTi
```

# 使用示例

```python
import cv2
from text_image_generator import Generator


gen = Generator(
    config_path="./config.yaml",
)

text = "𢞁𢞂𢞃𢞄𢞅"
text_with_font_list = gen.wrap_text_with_font_list(text)
img = gen.gen_image_from_text_with_font_list(
    text_with_font_list,
    text_color=(255, 105, 105),
    background_color=(166, 208, 221),
    apply_effect=False,
)

cv2.cvtColor(img, cv2.COLOR_RGB2BGR, dst=img)
cv2.imwrite("test.png", img)
```

生成圖片如下：

![示例圖片](./images/test.png)
