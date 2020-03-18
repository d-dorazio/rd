#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F32Range {
    low: f32,
    high: f32,
}

impl F32Range {
    pub const fn zero() -> Self {
        Self {
            low: 0.0,
            high: 0.0,
        }
    }

    pub const fn empty() -> Self {
        Self {
            low: std::f32::INFINITY,
            high: std::f32::NEG_INFINITY,
        }
    }

    pub fn expand(&mut self, v: f32) {
        self.low = self.low.min(v);
        self.high = self.high.max(v);
    }

    pub fn t(self, v: f32) -> f32 {
        (v - self.low) / (self.high - self.low)
    }
}
