mod f32range;

use f32range::F32Range;

#[derive(Debug, Clone)]
pub struct System {
    width: usize,
    height: usize,

    world: Vec<Cell>,
    world_buffer: Vec<Cell>,
    kernel: [f32; 9],

    pub feed_rate: f32,
    pub kill_rate: f32,
    pub diffusion_rates: (f32, f32),

    b_range: F32Range,
}

// chemical A, chemical B
pub type Cell = (f32, f32);

impl System {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;

        System {
            width,
            height,
            world: vec![(1.0, 0.0); size],
            world_buffer: vec![(1.0, 0.0); size],

            #[rustfmt::skip]
            kernel: [
                0.05,  0.20, 0.05,
                0.20, -1.00, 0.20,
                0.05,  0.20, 0.05,
            ],

            diffusion_rates: (1.0, 0.5),
            feed_rate: 0.055,
            kill_rate: 0.062,

            b_range: if size > 0 {
                F32Range::zero()
            } else {
                F32Range::empty()
            },
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Warning: be sure to call `update_metadata` after a call to `set` as it's not done
    /// automatically.
    pub fn set(&mut self, (x, y): (usize, usize), (a, b): Cell) {
        self.world[y * self.width + x] = (a, b);
    }

    /// Warning: slow operation, do it as few times as possible.
    pub fn update_metadata(&mut self) {
        self.b_range = F32Range::empty();
        for c in &self.world {
            self.b_range.expand(c.1);
        }
    }

    pub fn b_range(&self) -> F32Range {
        self.b_range
    }

    pub fn get(&self, (x, y): (usize, usize)) -> Cell {
        self.world[y * self.width + x]
    }

    pub fn cells(&self) -> impl Iterator<Item = ((usize, usize), Cell)> + '_ {
        self.world.iter().enumerate().map(move |(i, c)| {
            let x = i % self.width;
            let y = i / self.width;

            ((x, y), *c)
        })
    }

    /// Evolves the current state of the system
    ///
    /// It also updated the metadata because it's quite cheap to do here.
    #[allow(clippy::many_single_char_names)]
    pub fn evolve(&mut self, dt: f32) {
        let (da, db) = self.diffusion_rates;
        let f = self.feed_rate;
        let k = self.kill_rate;

        self.b_range = F32Range::empty();

        for (i, nc) in self.world_buffer.iter_mut().enumerate() {
            let (x, y) = (i % self.width, i / self.width);

            let lx = if x == 0 { self.width - 1 } else { x - 1 };
            let rx = (x + 1) % self.width;

            let ty = if y == 0 { self.height - 1 } else { y - 1 };
            let by = (y + 1) % self.height;

            #[rustfmt::skip]
            let neighbors = [
                (lx, ty), (x, ty), (rx, ty),
                (lx,  y), (x,  y), (rx,  y),
                (lx, by), (x, by), (rx, by),
            ];

            let mut neighbors_a = 0.0;
            let mut neighbors_b = 0.0;
            for ((xx, yy), k) in neighbors.iter().zip(&self.kernel) {
                let (a, b) = self.world[yy * self.width + xx];
                neighbors_a += k * a;
                neighbors_b += k * b;
            }

            let (a, b) = self.world[i];
            nc.0 = a + dt * (da * neighbors_a - a * b.powi(2) + f * (1.0 - a));
            nc.1 = b + dt * (db * neighbors_b + a * b.powi(2) - (k + f) * b);

            self.b_range.expand(nc.1);
        }

        std::mem::swap(&mut self.world, &mut self.world_buffer);
    }
}
