use cosmic_text::{Buffer, FontSystem, SwashCache};
use image::{GenericImage, GenericImageView, ImageBuffer};

pub fn generate_image(
    editor: &mut Buffer,
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    foreground_color: cosmic_text::Color,
    background_color: image::Rgb<u8>,
    width: usize,
    height: usize,
) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let mut raw_image = ImageBuffer::from_pixel(width as u32, height as u32, background_color);
    let mut right_border = 0;
    // Draw the buffer (for performance, instead use SwashCache directly)
    editor.draw(
        font_system,
        swash_cache,
        foreground_color,
        |x, y, _, _, color| {
            if x < 0 || x >= width as i32 || y < 0 || y >= height as i32 || (x == 0 && y == 0) {
                return;
            }
            if x > right_border {
                right_border = x
            }

            let (r, g, b, a) = (
                color.r() as u32,
                color.g() as u32,
                color.b() as u32,
                color.a() as u32,
            );
            let (raw_image_r, raw_image_g, raw_image_b) = unsafe {
                let tmp = raw_image.unsafe_get_pixel(x as u32, y as u32).0;
                (tmp[0] as u32, tmp[1] as u32, tmp[2] as u32)
            };
            let red = r * a / 255 + raw_image_r * (255 - a) / 255;
            let green = g * a / 255 + raw_image_g * (255 - a) / 255;
            let blue = b * a / 255 + raw_image_b * (255 - a) / 255;
            let rgb = image::Rgb([red as u8, green as u8, blue as u8]);

            unsafe {
                raw_image.unsafe_put_pixel(x as u32, y as u32, rgb);
            }
        },
    );

    raw_image
        .sub_image(0, 0, (right_border + 1) as u32, height as u32)
        .to_image()
}
