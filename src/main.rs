use std::convert::TryFrom;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use rand::prelude::*;
use structopt::StructOpt;

use rd::System;

/// Program to generate images and even videos showing
/// what Reaction Diffusion is all about.
#[derive(Debug, StructOpt)]
struct Opts {
    /// Width of the final image/video.
    #[structopt(short, long, default_value = "512")]
    width: u16,

    /// Height of the final image/video.
    #[structopt(short, long, default_value = "512")]
    height: u16,

    /// How many iterations of the simulation to run.
    ///
    /// A big number is suggested when the initial seed is small enough.
    #[structopt(short, long, default_value = "300")]
    iterations: usize,

    /// The rate chemical A is poured into the system.
    #[structopt(short, long, default_value = "0.055")]
    feed_rate: f32,

    /// The rate chemical B is killed from the system.
    #[structopt(short, long, default_value = "0.062")]
    kill_rate: f32,

    /// How much to speedup the image saving and the video.
    #[structopt(short, long, default_value = "1")]
    speed: usize,

    /// Whether to disable creating the video using ffmpeg or not.
    ///
    /// It's turned off by default if ffmpeg is not found.
    #[structopt(long)]
    without_video: bool,

    /// Where to store the temporary frames used to create the video.
    #[structopt(long, parse(from_os_str), default_value = "img")]
    img_dir: PathBuf,

    /// The seed to use to start the generation.
    #[structopt(subcommand)]
    seed: Option<Seed>,
}

#[derive(Debug, StructOpt)]
enum Seed {
    /// The process is seeded with a centered rectangle.
    /// Used mostly as an example.
    Rect,

    /// The process is seeded with a random grid of values.
    Random,

    /// The process is seeded with the given image that is automatically
    /// converted to grayscale.
    Image {
        #[structopt(parse(from_os_str))]
        input: PathBuf,
    },
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
    system.feed_rate = opts.feed_rate;
    system.kill_rate = opts.kill_rate;

    let width = system.width();
    let height = system.height();

    match &opts.seed {
        None | Some(Seed::Rect) => {
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
        Some(Seed::Random) => {
            let mut rng = thread_rng();
            for x in 0..width {
                for y in 0..height {
                    if rng.gen::<f32>() < 0.05 {
                        system.set((x, y), (1.0, 1.0));
                    }
                }
            }
        }
        Some(Seed::Image { input }) => {
            let im = image::open(input)
                .unwrap()
                .resize_exact(
                    opts.width.into(),
                    opts.height.into(),
                    image::imageops::FilterType::Gaussian,
                )
                .into_luma();

            for (x, y, p) in im.enumerate_pixels() {
                let g = p.0[0];
                if g < 127 {
                    system.set(
                        (usize::try_from(x).unwrap(), usize::try_from(y).unwrap()),
                        (1.0, 1.0),
                    );
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
        let with_video = !opts.without_video && Self::can_build_video();

        Self {
            img_dir: opts.img_dir.clone(),
            with_video,
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

    fn can_build_video() -> bool {
        match Command::new("ffmpeg")
            .args(&["-version"])
            .stdout(Stdio::null())
            .status()
        {
            Ok(e) if e.success() => true,
            Err(_) | Ok(_) => {
                eprintln!("disabling video output as ffmpeg was not found");
                false
            }
        }
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
