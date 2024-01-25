use conv::ValueInto;
use image::{GenericImage, GenericImageView, ImageBuffer, Pixel, Primitive};
use imageproc::{
    definitions::Clamp,
    drawing::{draw_hollow_rect_mut, Canvas},
    geometric_transformations::{Interpolation, Projection},
    rect::Rect,
};
use nalgebra::{Matrix3, Matrix4, Matrix4x2, Matrix4x3, SMatrix, SVector, Vector4};

type Matrix8 = SMatrix<f32, 8, 8>;
type Vector8 = SVector<f32, 8>;

/// Performs the perspective matrix transformation of vectors
///
/// ## Reference:
/// [OpenCV documentation](https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7)
pub fn perspective_transform(
    points: &Matrix4x3<f32>,
    transform_mat: &Matrix4<f32>,
) -> Matrix4x3<f32> {
    #[rustfmt::skip]
    let points_pad_one: Matrix4<f32> = Matrix4::new(
        points.m11, points.m21, points.m31, points.m41,
        points.m12, points.m22, points.m32, points.m42,
        points.m13, points.m23, points.m33, points.m43,
        1., 1., 1., 1.,
    );

    let mul: Matrix4<f32> = transform_mat * points_pad_one;

    let row0: Vector4<f32> = mul.column(0) / mul.m41;
    let row1: Vector4<f32> = mul.column(1) / mul.m42;
    let row2: Vector4<f32> = mul.column(2) / mul.m43;
    let row3: Vector4<f32> = mul.column(3) / mul.m44;

    #[rustfmt::skip]
    let res = Matrix4x3::new(
        row0.x, row0.y, row0.z,
        row1.x, row1.y, row1.z,
        row2.x, row2.y, row2.z,
        row3.x, row3.y, row3.z,
    );

    res
}

/// ## Reference:
/// [OpenCV implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/imgwarp.cpp#L3408-L3459)
pub fn get_perspective_transform(
    points_in: &Matrix4x2<f32>,
    points_out: &Matrix4x2<f32>,
) -> Matrix3<f32> {
    #[rustfmt::skip]
    let left = Matrix8::from_vec(vec![
        points_in.m11, points_in.m12, 1., 0., 0., 0., -points_in.m11 * points_out.m11, -points_in.m12 * points_out.m11,
        points_in.m21, points_in.m22, 1., 0., 0., 0., -points_in.m21 * points_out.m21, -points_in.m22 * points_out.m21,
        points_in.m31, points_in.m32, 1., 0., 0., 0., -points_in.m31 * points_out.m31, -points_in.m32 * points_out.m31,
        points_in.m41, points_in.m42, 1., 0., 0., 0., -points_in.m41 * points_out.m41, -points_in.m42 * points_out.m41,
        0., 0., 0., points_in.m11, points_in.m12, 1., -points_in.m11 * points_out.m12, -points_in.m12 * points_out.m12,
        0., 0., 0., points_in.m21, points_in.m22, 1., -points_in.m21 * points_out.m22, -points_in.m22 * points_out.m22,
        0., 0., 0., points_in.m31, points_in.m32, 1., -points_in.m31 * points_out.m32, -points_in.m32 * points_out.m32,
        0., 0., 0., points_in.m41, points_in.m42, 1., -points_in.m41 * points_out.m42, -points_in.m42 * points_out.m42,
    ]).transpose();

    #[rustfmt::skip]
    let right  = Vector8::from_vec(vec![
        points_out.m11,
        points_out.m21,
        points_out.m31,
        points_out.m41,
        points_out.m12,
        points_out.m22,
        points_out.m32,
        points_out.m42,
    ]);

    let decomp = left.lu();
    let x = decomp.solve(&right).expect("Linear resolution failed.");

    unsafe {
        Matrix3::new(
            *x.get_unchecked(0),
            *x.get_unchecked(1),
            *x.get_unchecked(2),
            *x.get_unchecked(3),
            *x.get_unchecked(4),
            *x.get_unchecked(5),
            *x.get_unchecked(6),
            *x.get_unchecked(7),
            1.0,
        )
    }
}

pub fn warp_perspective<I, P, S>(
    src: &I,
    transform_mat: &Matrix3<f32>,
    side_length: u32,
    default: P,
) -> ImageBuffer<P, Vec<S>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S> + 'static + Sync + Send,
    S: Primitive + 'static + Sync + Send + ValueInto<f32> + Clamp<f32>,
{
    #[rustfmt::skip]
    let projection = Projection::from_matrix([
        transform_mat.m11, transform_mat.m12, transform_mat.m13,
        transform_mat.m21, transform_mat.m22, transform_mat.m23,
        transform_mat.m31, transform_mat.m32, transform_mat.m33, 
    ]).unwrap();

    let mut padded_image = ImageBuffer::from_pixel(side_length, side_length, default);
    padded_image.copy_from(src, 0, 0).unwrap();

    imageproc::geometric_transformations::warp(
        &padded_image,
        &projection,
        Interpolation::Bilinear,
        default,
    )
}

/// Draws the outline of a rectangle on an image in place.
///
/// Draws as much of the boundary of the rectangle as lies inside the image bounds.
pub fn rectangle<C>(canvas: &mut C, rect: Rect, color: C::Pixel, thickness: u32)
where
    C: Canvas,
{
    let left = rect.left();
    let right = rect.right();
    let top = rect.top();
    let bottom = rect.bottom();

    draw_hollow_rect_mut(
        canvas,
        Rect::at(left, top).of_size((right - left + 1) as u32, thickness),
        color,
    );
    draw_hollow_rect_mut(
        canvas,
        Rect::at(left, bottom).of_size((right - left + 1) as u32, thickness),
        color,
    );
    draw_hollow_rect_mut(
        canvas,
        Rect::at(left, top).of_size(thickness, (bottom - top + 1) as u32),
        color,
    );
    draw_hollow_rect_mut(
        canvas,
        Rect::at(right, top).of_size(thickness, (bottom - top + 1) as u32),
        color,
    );
}
