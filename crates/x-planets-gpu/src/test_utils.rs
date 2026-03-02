//! Test utilities for GPU code.
//!
//! Karpathy principle: CPU is the ground truth.
//! Everything the GPU does must be verifiable against a CPU reference.

/// Generate a checkerboard RGBA texture (CPU).
/// Useful as a test pattern for verifying texture upload/sampling.
pub fn checkerboard_rgba(width: u32, height: u32, cell_size: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let cx = x / cell_size;
            let cy = y / cell_size;
            let is_white = (cx + cy).is_multiple_of(2);

            if is_white {
                pixels.extend_from_slice(&[255, 255, 255, 255]);
            } else {
                pixels.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
    }

    pixels
}

/// Generate a solid color RGBA texture.
pub fn solid_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel = [r, g, b, a];
    pixel.repeat((width * height) as usize)
}

/// Generate a gradient RGBA texture (red increases left→right, green top→bottom).
pub fn gradient_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let r = (x as f32 / width as f32 * 255.0) as u8;
            let g = (y as f32 / height as f32 * 255.0) as u8;
            pixels.extend_from_slice(&[r, g, 0, 255]);
        }
    }

    pixels
}

/// Compare two RGBA pixel buffers. Returns (max_channel_diff, mean_channel_diff).
pub fn pixel_diff(a: &[u8], b: &[u8]) -> PixelDiff {
    assert_eq!(a.len(), b.len(), "pixel buffers must be same size");

    let mut max_diff: u8 = 0;
    let mut sum_diff: u64 = 0;

    for (pa, pb) in a.iter().zip(b.iter()) {
        let d = (*pa as i16 - *pb as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }

    PixelDiff {
        max_channel_diff: max_diff,
        mean_channel_diff: sum_diff as f64 / a.len() as f64,
        total_pixels: a.len() / 4,
    }
}

#[derive(Debug)]
pub struct PixelDiff {
    pub max_channel_diff: u8,
    pub mean_channel_diff: f64,
    pub total_pixels: usize,
}

impl PixelDiff {
    /// Are the images identical?
    pub fn is_exact(&self) -> bool {
        self.max_channel_diff == 0
    }

    /// Are the images "close enough" for GPU float precision?
    pub fn is_close(&self, tolerance: u8) -> bool {
        self.max_channel_diff <= tolerance
    }

    /// Compute PSNR (Peak Signal-to-Noise Ratio) in dB.
    /// Higher is better. >40 dB is visually identical.
    pub fn psnr(&self) -> f64 {
        if self.mean_channel_diff == 0.0 {
            return f64::INFINITY;
        }
        let mse = self.mean_channel_diff * self.mean_channel_diff;
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

impl std::fmt::Display for PixelDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_diff={} mean_diff={:.4} psnr={:.1}dB pixels={}",
            self.max_channel_diff,
            self.mean_channel_diff,
            self.psnr(),
            self.total_pixels,
        )
    }
}

/// Save raw RGBA pixels as a PNG file (for visual inspection).
pub fn save_rgba_png(path: &std::path::Path, pixels: &[u8], width: u32, height: u32) {
    use image::{ImageBuffer, Rgba};
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(width, height, pixels.to_vec())
            .expect("pixel buffer size mismatch");
    img.save(path).expect("failed to save PNG");
}

// ═══════════════════════════════════════════════════════════════════
// Tile label texture
// ═══════════════════════════════════════════════════════════════════

/// 5×7 pixel bitmap font: indices 0-9 = '0'-'9', index 10 = '/'.
/// Each entry is 7 rows; each row is a 5-bit mask (bit 4 = leftmost pixel).
const FONT_5X7: [[u8; 7]; 11] = [
    [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], // 0
    [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E], // 1
    [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F], // 2
    [0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E], // 3
    [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02], // 4
    [0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E], // 5
    [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E], // 6
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08], // 7
    [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E], // 8
    [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C], // 9
    [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10], // /
];

fn font_glyph(c: char) -> Option<&'static [u8; 7]> {
    match c {
        '0'..='9' => Some(&FONT_5X7[(c as usize) - ('0' as usize)]),
        '/' => Some(&FONT_5X7[10]),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_glyph(pixels: &mut [u8], stride: u32, glyph: &[u8; 7], ox: u32, oy: u32, scale: u32, fg: [u8; 4], total_h: u32) {
    for (row, &bits) in glyph.iter().enumerate() {
        for col in 0u32..5 {
            if bits & (1 << (4 - col)) != 0 {
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = ox + col * scale + sx;
                        let py = oy + row as u32 * scale + sy;
                        if py < total_h {
                            let idx = ((py * stride + px) * 4) as usize;
                            if idx + 3 < pixels.len() {
                                pixels[idx..idx + 4].copy_from_slice(&fg);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Generate a checkerboard tile texture with "{z}/{x}/{y}" label.
///
/// Suitable for debug rendering to visually identify each tile.
pub fn tile_label_rgba(width: u32, height: u32, z: u32, x: u32, y: u32) -> Vec<u8> {
    let mut pixels = checkerboard_rgba(width, height, 32);

    let label = format!("{}/{}/{}", z, x, y);
    let scale: u32 = (width / 128).clamp(1, 4);
    let char_w = (5 + 1) * scale;
    let char_h = 7 * scale;

    // Dark blue text with background
    let text_w = label.len() as u32 * char_w;
    let text_h = char_h + 2 * scale;

    let ox = (width.saturating_sub(text_w)) / 2;
    let oy = (height.saturating_sub(text_h)) / 2;

    // Draw semi-transparent dark background strip
    let bg: [u8; 4] = [20, 20, 60, 200];
    for by in 0..text_h {
        for bx in 0..text_w {
            let px = ox + bx;
            let py = oy + by;
            if px < width && py < height {
                let idx = ((py * width + px) * 4) as usize;
                if idx + 3 < pixels.len() {
                    pixels[idx..idx + 4].copy_from_slice(&bg);
                }
            }
        }
    }

    // Draw text on top
    let fg: [u8; 4] = [255, 240, 50, 255];
    for (i, c) in label.chars().enumerate() {
        if let Some(glyph) = font_glyph(c) {
            let gx = ox + i as u32 * char_w;
            let gy = oy + scale;
            draw_glyph(&mut pixels, width, glyph, gx, gy, scale, fg, height);
        }
    }

    pixels
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkerboard_dimensions() {
        let pixels = checkerboard_rgba(8, 8, 2);
        assert_eq!(pixels.len(), 8 * 8 * 4);

        // top-left pixel should be white
        assert_eq!(&pixels[0..4], &[255, 255, 255, 255]);

        // pixel at (2,0) should be black (next cell)
        let offset = 2 * 4;
        assert_eq!(&pixels[offset..offset + 4], &[0, 0, 0, 255]);
    }

    #[test]
    fn test_solid_color() {
        let pixels = solid_rgba(4, 4, 128, 64, 32, 255);
        assert_eq!(pixels.len(), 4 * 4 * 4);
        assert_eq!(&pixels[0..4], &[128, 64, 32, 255]);
        assert_eq!(&pixels[60..64], &[128, 64, 32, 255]); // last pixel
    }

    #[test]
    fn test_pixel_diff_identical() {
        let a = solid_rgba(4, 4, 100, 100, 100, 255);
        let b = solid_rgba(4, 4, 100, 100, 100, 255);
        let diff = pixel_diff(&a, &b);
        assert!(diff.is_exact());
        assert_eq!(diff.psnr(), f64::INFINITY);
    }

    #[test]
    fn test_pixel_diff_small_difference() {
        let a = solid_rgba(4, 4, 100, 100, 100, 255);
        let b = solid_rgba(4, 4, 101, 100, 100, 255);
        let diff = pixel_diff(&a, &b);
        assert!(!diff.is_exact());
        assert!(diff.is_close(1));
        assert!(diff.psnr() > 40.0); // visually identical
    }

    #[test]
    fn test_pixel_diff_large_difference() {
        let a = solid_rgba(4, 4, 0, 0, 0, 255);
        let b = solid_rgba(4, 4, 255, 255, 255, 255);
        let diff = pixel_diff(&a, &b);
        assert!(!diff.is_close(100));
        assert!(diff.psnr() < 10.0); // very different
    }

    #[test]
    fn test_gradient_range() {
        let pixels = gradient_rgba(256, 256);
        assert_eq!(pixels.len(), 256 * 256 * 4);

        // top-left: r=0, g=0
        assert_eq!(pixels[0], 0); // r
        assert_eq!(pixels[1], 0); // g

        // bottom-right: r≈255, g≈255
        let last = (255 * 256 + 255) * 4;
        assert!(pixels[last] > 250);     // r
        assert!(pixels[last + 1] > 250); // g
    }
}
