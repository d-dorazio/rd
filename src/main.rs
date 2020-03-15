use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use rand::prelude::*;

#[derive(Debug, Clone)]
struct System {
    width: usize,
    height: usize,

    world: Vec<Cell>,
    world_buffer: Vec<Cell>,
    kernel: [f32; 9],

    pub feed_rate: f32,
    pub kill_rate: f32,
    pub diffusion_rates: (f32, f32),
}

// chemical A, chemical B
type Cell = (f32, f32);

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
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn set(&mut self, (x, y): (usize, usize), (a, b): Cell) {
        self.world[y * self.width + x] = (a, b);
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

    #[allow(clippy::many_single_char_names)]
    pub fn evolve(&mut self, dt: f32) {
        let (da, db) = self.diffusion_rates;
        let f = self.feed_rate;
        let k = self.kill_rate;

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
        }

        std::mem::swap(&mut self.world, &mut self.world_buffer);
    }
}

fn main() {
    let (width, height) = (1024_u16, 1024_u16);
    let iterations = 3600;
    let framerate = 120;
    let img_dir = Path::new("img");
    let video = true;
    let random_starting_state = false;

    if !img_dir.exists() {
        fs::create_dir(img_dir).unwrap();
    }

    for entry in img_dir.read_dir().unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().ends_with(".png") {
            fs::remove_file(entry.path()).unwrap();
        }
    }

    let start_ts = Instant::now();
    let mut system = System::new(width.into(), height.into());

    {
        let width = system.width();
        let height = system.height();

        if random_starting_state {
            let mut rng = thread_rng();
            for x in 0..width {
                for y in 0..height {
                    if rng.gen::<f32>() < 0.05 {
                        system.set((x, y), (1.0, 1.0));
                    }
                }
            }
        } else {
            let l = width.min(height) / 4;
            let ty = height / 2 - l / 2;
            let sx = width / 2 - l / 2;
            let by = ty + l;
            let rx = sx + l;

            for i in 0..l {
                system.set((sx + i, ty), (1.0, 1.0));
                system.set((sx + i, by), (1.0, 1.0));
                system.set((sx, ty + i), (1.0, 1.0));
                system.set((rx, ty + i), (1.0, 1.0));
            }
        }
    }

    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    let mut img = image::GrayImage::new(width.into(), height.into());
    let mut render = |system: &System, path: &OsStr| {
        for ((_, c), pix) in system.cells().zip(img.pixels_mut()) {
            *pix = image::Luma([(c.1 * 255.0) as u8]);
        }

        img.save(path).unwrap();
    };

    for i in 0..iterations {
        write!(stdout, "\r iteration: {}", i).unwrap();
        stdout.flush().unwrap();

        system.evolve(1.0);

        if video {
            render(&system, img_dir.join(&format!("rd-{}.png", i)).as_os_str());
        }
    }

    let elapsed = start_ts.elapsed();

    if video {
        Command::new("ffmpeg")
            .args(&[
                "-framerate",
                &framerate.to_string(),
                "-i",
                img_dir.join("rd-%00d.png").to_str().unwrap(),
                "-pix_fmt",
                "yuv420p",
                "-y",
                "rd.mp4",
            ])
            .status()
            .unwrap();
    } else {
        render(&system, OsStr::new("rd.png"));
    }

    writeln!(
        stdout,
        r#"
*** Results ***

generation took {} min {} secs
"#,
        elapsed.as_secs() / 60,
        elapsed.as_secs() % 60
    )
    .unwrap();
}