use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use rand::prelude::*;
use structopt::StructOpt;

use rd::System;

#[derive(Debug, StructOpt)]
struct Opts {
    #[structopt(short, long, default_value = "512")]
    width: u16,

    #[structopt(short, long, default_value = "512")]
    height: u16,

    #[structopt(short, long, default_value = "300")]
    iterations: usize,

    #[structopt(short, long, default_value = "1")]
    speed: usize,

    #[structopt(long)]
    without_video: bool,

    #[structopt(long, parse(from_os_str), default_value = "img")]
    img_dir: PathBuf,

    #[structopt(subcommand)]
    mode: Option<Mode>,
}

#[derive(Debug, StructOpt)]
enum Mode {
    Rect,
    Random,
}

struct Renderer {
    with_video: bool,
    speed: usize,
    img_dir: PathBuf,
    tmp_img: image::GrayImage,
}

fn main() {
    let opts = Opts::from_args();

    setup_img_dir(&opts);

    let start_ts = Instant::now();

    let mut system = create_system(&opts);

    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    let mut renderer = Renderer::new(&opts);
    renderer.start(&system);

    for i in 1..=opts.iterations {
        write!(stdout, "\riteration: {}", i).unwrap();
        stdout.flush().unwrap();

        system.evolve(1.0);

        renderer.snapshot(&system, i);
    }

    let elapsed = start_ts.elapsed();

    renderer.end(&system);

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

fn create_system(opts: &Opts) -> System {
    let mut system = System::new(opts.width.into(), opts.height.into());

    let width = system.width();
    let height = system.height();

    match opts.mode {
        None | Some(Mode::Rect) => {
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
        Some(Mode::Random) => {
            let mut rng = thread_rng();
            for x in 0..width {
                for y in 0..height {
                    if rng.gen::<f32>() < 0.05 {
                        system.set((x, y), (1.0, 1.0));
                    }
                }
            }
        }
    }

    system.update_metadata();

    system
}

fn setup_img_dir(opts: &Opts) {
    if !opts.img_dir.exists() {
        fs::create_dir(&opts.img_dir).unwrap();
    }

    for entry in opts.img_dir.read_dir().unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().ends_with(".png") {
            fs::remove_file(entry.path()).unwrap();
        }
    }
}

impl Renderer {
    fn new(opts: &Opts) -> Self {
        Self {
            img_dir: opts.img_dir.clone(),
            with_video: !opts.without_video,
            speed: opts.speed,

            tmp_img: image::GrayImage::new(opts.width.into(), opts.height.into()),
        }
    }

    fn start(&mut self, system: &System) {
        if self.with_video {
            let path = self.img_dir.join("rd-0.png");
            self.render_frame(system, &path);
        }
    }

    fn snapshot(&mut self, system: &System, gen: usize) {
        assert!(gen > 0);

        if self.with_video && gen % self.speed == 0 {
            let path = self.img_dir.join(&format!("rd-{}.png", gen / self.speed));
            self.render_frame(&system, &path);
        }
    }

    fn end(&mut self, system: &System) {
        self.render_frame(&system, Path::new("rd.png"));

        if self.with_video {
            self.build_video();
        }
    }

    fn render_frame(&mut self, system: &System, path: &Path) {
        for ((_, c), pix) in system.cells().zip(self.tmp_img.pixels_mut()) {
            let g = system.b_range().t(c.1) * 255.0;
            *pix = image::Luma([g as u8]);
        }

        self.tmp_img.save(path).unwrap();
    }

    fn build_video(&self) {
        Command::new("ffmpeg")
            .args(&[
                "-framerate",
                "60",
                "-i",
                self.img_dir.join("rd-%00d.png").to_str().unwrap(),
                "-pix_fmt",
                "yuv420p",
                "-y",
                "rd.mp4",
            ])
            .status()
            .unwrap();
    }
}
