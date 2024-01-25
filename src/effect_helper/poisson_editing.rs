use core::ops::{AddAssign, Div, Neg, Sub};

use image::GrayImage;
use nalgebra::DMatrix;

/// The main reference here is the Numba-based Solver implementation
pub struct Solver {
    mask: DMatrix<f64>,
    mask_not: DMatrix<f64>,
    target: DMatrix<f64>,
    grad: DMatrix<f64>,
}

impl Solver {
    pub fn reset(mask: DMatrix<f64>, target: DMatrix<f64>, grad: DMatrix<f64>) -> Self {
        let mask_not = mask.add_scalar(-1.0).neg();
        let mut target = target;

        let tmp = Self::grid_iter(&grad, &target);
        // tgt[self.bool_mask] = tmp[self.bool_mask] / 4.0
        target.component_mul_assign(&mask_not);
        target.add_assign(tmp.component_mul(&mask).div(4.0));

        Self {
            mask,
            mask_not,
            target,
            grad,
        }
    }

    pub fn step(&mut self, iteration: usize) -> (DMatrix<u8>, f64) {
        for _ in 0..iteration {
            let target = Self::grid_iter(&self.grad, &self.target);
            // self.tgt[self.bool_mask] = tgt[self.bool_mask] / 4.0
            self.target.component_mul_assign(&self.mask_not);
            self.target
                .add_assign(target.component_mul(&self.mask).div(4.0));
        }

        let mut tmp = (&self.target * 4.0).sub(&self.grad);
        let (tmp_height, tmp_width) = tmp.shape();
        let (target_height, target_width) = self.target.shape();
        // tmp[1:] -= self.tgt[:-1]
        tmp.view_range_mut(1.., ..)
            .add_assign(self.target.view_range(..(target_height - 1), ..).neg());
        // tmp[:-1] -= self.tgt[1:]
        tmp.view_range_mut(..(tmp_height - 1), ..)
            .add_assign(self.target.view_range(1.., ..).neg());
        // tmp[:, 1:] -= self.tgt[:, :-1]
        tmp.view_range_mut(.., 1..)
            .add_assign(self.target.view_range(.., ..(target_width - 1)).neg());
        // tmp[:, :-1] -= self.tgt[:, 1:]
        tmp.view_range_mut(.., ..(tmp_width - 1))
            .add_assign(self.target.view_range(.., 1..).neg());

        let err = tmp.component_mul(&self.mask).abs().sum();

        (
            // Matrix::from_iterator is column-major
            // Matrix::iter is also column-major
            DMatrix::from_iterator(
                self.target.nrows(),
                self.target.ncols(),
                self.target.iter().map(|each| each.clamp(0.0, 255.0) as u8),
            ),
            err,
        )
    }

    fn grid_iter(grad: &DMatrix<f64>, target: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = grad.clone();
        let (result_height, result_width) = result.shape();
        let (target_height, target_width) = target.shape();
        // result[1:] += target[:-1]
        result
            .view_range_mut(1.., ..)
            .add_assign(target.view_range(..(target_height - 1), ..));
        // result[:-1] += target[1:]
        result
            .view_range_mut(..(result_height - 1), ..)
            .add_assign(target.view_range(1.., ..));
        // result[:, 1:] += target[:, :-1]
        result
            .view_range_mut(.., 1..)
            .add_assign(target.view_range(.., ..(target_width - 1)));
        // result[:, :-1] += target[:, 1:]
        result
            .view_range_mut(.., ..(result_width - 1))
            .add_assign(target.view_range(.., 1..));

        result
    }
}

pub enum Gradient {
    Maximum,
    Source,
    Average,
}

macro_rules! mix_grad {
    ( $a:expr, $b:expr , $gradient: expr ) => {
        match $gradient {
            Gradient::Average => ($a + $b).div(2.0),
            Gradient::Source => $a.into(),
            Gradient::Maximum => $a.zip_map(&$b, |a, b| if a.abs() >= b.abs() { a } else { b }),
        }
    };
}

/// This is a port of the project [Fast-Poisson-Image-Editing](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing)
pub struct Processor {
    // gradient: Gradient,
    solver: Solver,
    target: DMatrix<u8>,
    target_cord: (usize, usize, usize, usize), // x0, y0, x1, y1
}

impl Processor {
    pub fn reset(
        source: GrayImage,
        mask: GrayImage,
        target: GrayImage,
        mask_on_source: (usize, usize),
        mask_on_target: (usize, usize),
        gradient: Gradient,
    ) -> Self {
        let source = DMatrix::from_row_iterator(
            source.height() as usize,
            source.width() as usize,
            source.into_vec().into_iter().map(|each| each as f64),
        );
        let target = DMatrix::from_row_iterator(
            target.height() as usize,
            target.width() as usize,
            target.into_vec().into_iter().map(|each| each as f64),
        );
        let [mask_height, mask_width] = [mask.height() as usize, mask.width() as usize];
        let mut mask = DMatrix::from_row_iterator(
            mask_height,
            mask_width,
            mask.into_vec()
                .into_iter()
                .map(|each| if each >= 128 { 1.0 } else { 0.0 }),
        );
        // mask[0] = 0
        mask.row_mut(0).apply(|each| *each = 0.0);
        // mask[-1] = 0
        mask.row_mut(mask_height - 1).apply(|each| *each = 0.0);
        // mask[:, 0] = 0
        mask.column_mut(0).apply(|each| *each = 0.0);
        // mask[:, -1] = 0
        mask.column_mut(mask_width - 1).apply(|each| *each = 0.0);

        let (mut x0, mut y0, mut x1, mut y1) = Self::get_border(&mask);
        (x0, y0, x1, y1) = (x0 - 1, y0 - 1, x1 + 2, y1 + 2);
        let mask = mask.view_range(y0..y1, x0..x1);

        let source_crop = source.view_range(
            (mask_on_source.1 + y0)..(mask_on_source.1 + y1),
            (mask_on_source.0 + x0)..(mask_on_source.0 + x1),
        );
        let target_crop = target.view_range(
            (mask_on_target.1 + y0)..(mask_on_target.1 + y1),
            (mask_on_target.0 + x0)..(mask_on_target.0 + x1),
        );

        let mut grad = DMatrix::zeros(mask.nrows(), mask.ncols());
        grad.view_range_mut(1.., ..).add_assign(mix_grad!(
            source_crop.view_range(1.., ..)
                - source_crop.view_range(..(source_crop.nrows() - 1), ..),
            target_crop.view_range(1.., ..)
                - target_crop.view_range(..(target_crop.nrows() - 1), ..),
            gradient
        ));
        grad.view_range_mut(..(grad.nrows() - 1), ..)
            .add_assign(mix_grad!(
                source_crop.view_range(..(source_crop.nrows() - 1), ..)
                    - source_crop.view_range(1.., ..),
                target_crop.view_range(..(target_crop.nrows() - 1), ..)
                    - target_crop.view_range(1.., ..),
                gradient
            ));
        grad.view_range_mut(.., 1..).add_assign(mix_grad!(
            source_crop.view_range(.., 1..)
                - source_crop.view_range(.., ..(source_crop.ncols() - 1)),
            target_crop.view_range(.., 1..)
                - target_crop.view_range(.., ..(target_crop.ncols() - 1)),
            gradient
        ));
        grad.view_range_mut(.., ..(grad.ncols() - 1))
            .add_assign(mix_grad!(
                source_crop.view_range(.., ..(source_crop.ncols() - 1))
                    - source_crop.view_range(.., 1..),
                target_crop.view_range(.., ..(target_crop.ncols() - 1))
                    - target_crop.view_range(.., 1..),
                gradient
            ));
        grad.component_mul_assign(&mask);

        let target_cord = (
            mask_on_target.0 + x0,
            mask_on_target.0 + x1,
            mask_on_target.1 + y0,
            mask_on_target.1 + y1,
        );
        let solver = Solver::reset(mask.into(), target_crop.into(), grad);

        Self {
            // gradient,
            solver,
            target: nalgebra::try_convert(target).unwrap(),
            target_cord,
        }
    }

    pub fn step(&mut self, iteration: usize) -> (DMatrix<u8>, f64) {
        let (target, err) = self.solver.step(iteration);
        let (x0, x1, y0, y1) = self.target_cord;

        self.target
            .view_range_mut(y0..y1, x0..x1)
            .copy_from(&target);
        return (self.target.clone(), err);
    }

    fn get_border(mat: &DMatrix<f64>) -> (usize, usize, usize, usize) {
        let [height, width] = [mat.nrows(), mat.ncols()];

        let [mut x0, mut y0, mut x1, mut y1] = [0, 0, 0, 0];
        for x in 0..width {
            if mat.column(x).sum() > 0.0 {
                x0 = x;
                break;
            }
        }
        for y in 0..height {
            if mat.row(y).sum() > 0.0 {
                y0 = y;
                break;
            }
        }

        if x0 == width - 1 {
            x1 = width - 1
        } else {
            for x in (0..width).rev() {
                if mat.column(x).sum() > 0.0 {
                    x1 = x;
                    break;
                }
            }
        }

        if y0 == height - 1 {
            y1 = height - 1
        } else {
            for y in (0..height).rev() {
                if mat.row(y).sum() > 0.0 {
                    y1 = y;
                    break;
                }
            }
        }

        (x0, y0, x1, y1)
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_pie() {
        let start = Instant::now();
        let source = image::open("./test-img/source.jpg").unwrap();
        let mask = image::open("./test-img/mask.jpg").unwrap();
        let background = image::open("./test-img/background.jpg").unwrap();
        let source = image::imageops::grayscale(&source);
        let mask = image::imageops::grayscale(&mask);
        let background = image::imageops::grayscale(&background);

        let mut processor = Processor::reset(
            source,
            mask,
            background,
            (0, 0),
            (100, 100),
            Gradient::Maximum,
        );
        let (target, _) = processor.step(5000);

        let res = GrayImage::from_vec(
            target.ncols() as u32,
            target.nrows() as u32,
            target.transpose().iter().map(|&each| each).collect(),
        )
        .unwrap();
        res.save("./test-img/pie.png").unwrap();
        println!("{}", start.elapsed().as_secs_f64());
    }
}
