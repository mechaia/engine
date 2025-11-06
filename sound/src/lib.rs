pub mod mix;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Sample, SampleRate, SupportedBufferSize, SupportedStreamConfigRange,
};
use std::{io, num::NonZeroUsize, sync::Arc};
use symphonia::{
    core::{
        audio::{AudioBuffer, AudioBufferRef, Channels, Signal},
        codecs::Decoder,
        conv::FromSample,
        formats::{FormatOptions, FormatReader},
        io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions},
        meta::MetadataOptions,
        sample::Sample as SymSample,
    },
    default::{get_codecs, get_probe},
};
use util::math::fixed::{U0d32, U32d32};
use util::sync::spsc;

pub struct Dev {
    host: cpal::Host,
    device: cpal::Device,
    stream: cpal::Stream,
    config: cpal::StreamConfig,
}

pub struct DevConfig {
    pub channels: u16,
    pub sample_rate: u32,
    pub buffer_size: BufferSize,
}

pub enum BufferSize {
    Default,
    Fixed(u32),
    MaxLatency(U0d32),
}

pub struct AudioData {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    ticker: U32d32,
}

impl Dev {
    pub fn new<R: Send + FnMut(&mut [f32], u16, U0d32) + 'static>(
        config: &DevConfig,
        mut reader: R,
    ) -> Self {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .expect("no output device available");

        let mut supported_configs_ranges = device
            .supported_output_configs()
            .expect("error while querying configs");

        let supported_cfg = {
            let f = |range: SupportedStreamConfigRange| {
                if config.sample_rate < range.min_sample_rate().0 {
                    range.with_sample_rate(range.min_sample_rate())
                } else if config.sample_rate < range.max_sample_rate().0 {
                    range.with_sample_rate(SampleRate(config.sample_rate))
                } else {
                    range.with_max_sample_rate()
                }
            };

            let mut cfg = f(supported_configs_ranges.next().unwrap());

            for range in supported_configs_ranges {
                if cfg.channels() == config.channels && cfg.sample_rate().0 == config.sample_rate {
                    break;
                }
                if config.channels != range.channels() && config.channels == cfg.channels() {
                    continue;
                }
                cfg = f(range);
            }

            cfg
        };

        let buffer_size = 'buffer_size: {
            match supported_cfg.buffer_size() {
                &SupportedBufferSize::Range { min, max } => {
                    let n = match config.buffer_size {
                        BufferSize::Default => break 'buffer_size cpal::BufferSize::Default,
                        BufferSize::Fixed(n) => n,
                        BufferSize::MaxLatency(x) => {
                            (U32d32::from(x) * supported_cfg.sample_rate().0).int()
                        }
                    };
                    #[cfg(target_os = "linux")]
                    if host.id() == cpal::HostId::Alsa {
                        // fix retardedness
                        let n = n.next_power_of_two();
                        break 'buffer_size cpal::BufferSize::Fixed(n.clamp(min, max));
                    }
                    cpal::BufferSize::Fixed(n.clamp(min, max))
                }
                SupportedBufferSize::Unknown => cpal::BufferSize::Default,
            }
        };

        let mut cfg = supported_cfg.config();
        cfg.buffer_size = buffer_size;

        let channel_count = cfg.channels;

        let dt = (U32d32::ONE / cfg.sample_rate.0).frac();

        let err_fn = move |err| eprintln!("an error occurred on the output audio stream: {}", err);

        let stream = device
            .build_output_stream(
                &cfg,
                move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    let t = _info.timestamp();
                    reader(data, channel_count, dt)
                },
                err_fn,
                None,
            )
            .unwrap();

        Self {
            host,
            device,
            stream,
            config: cfg,
        }
    }

    pub fn play(&mut self) {
        self.stream.play().unwrap();
    }

    pub fn pause(&mut self) {
        self.stream.pause().unwrap();
    }

    /// Audio sample rate, in Hertz.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate.0
    }

    /// Audio sample rate, in Hertz.
    pub fn sample_rate_f32(&self) -> f32 {
        self.config.sample_rate.0 as f32
    }

    pub fn frame_time(&self) -> U32d32 {
        U32d32::ONE / self.config.sample_rate.0
    }

    /// Audio channels
    pub fn channels(&self) -> u16 {
        self.config.channels
    }
}

impl AudioData {
    fn new(data: Arc<Box<[u8]>>) -> Self {
        let source = make_source(data);
        let res = get_probe()
            .format(
                &Default::default(),
                source,
                &FormatOptions {
                    ..Default::default()
                },
                &MetadataOptions {
                    ..Default::default()
                },
            )
            .unwrap();
        let track = res.format.default_track().expect("no tracks");
        let mut params = track.codec_params.clone();
        params.channels = Some(Channels::FRONT_LEFT | Channels::FRONT_RIGHT);
        params.channel_layout = Some(symphonia::core::audio::Layout::Stereo);
        let decoder = get_codecs().make(&params, &Default::default()).unwrap();

        Self {
            format: res.format,
            decoder,
            ticker: U32d32::ONE,
        }
    }

    pub fn channels(&self) -> u16 {
        self.decoder.last_decoded().spec().channels.count() as _
    }

    /// Returns `false` if at end of stream.
    fn take(&mut self, buf: &mut Vec<f32>, mut count: usize, dt: U32d32) -> bool {
        macro_rules! each {
            ($($ty:ident)*) => {
                match self.decoder.last_decoded() {
                    $(AudioBufferRef::$ty(b) => Self::take_ref(&mut self.ticker, &*b, buf, count, dt),)*
                }
            };
        }
        while count > 0 {
            let (res, has_more) = each!(U8 U16 U24 U32 S8 S16 S24 S32 F32 F64);
            count -= res;
            if !has_more {
                let packet = match self.format.next_packet() {
                    Ok(p) => p,
                    Err(symphonia::core::errors::Error::IoError(e))
                        if e.kind() == io::ErrorKind::UnexpectedEof =>
                    {
                        return false
                    }
                    Err(e) => todo!("{e}"),
                };
                self.decoder.decode(&packet).unwrap();
            }
        }
        true
    }

    /// Returns `false` if out of frames.
    /// Rewinds ticker when out of frames.
    fn take_ref<T: SymSample>(
        ticker: &mut U32d32,
        from: &AudioBuffer<T>,
        to: &mut Vec<f32>,
        max_count: usize,
        dt: U32d32,
    ) -> (usize, bool)
    where
        f32: FromSample<T>,
    {
        let dt = dt * from.spec().rate;
        let channels = from.spec().channels.count();

        let og_len = to.len();

        for c in 0..max_count {
            if ticker.int() as usize >= from.frames() {
                *ticker -= U32d32::from(from.frames() as u32);
                return (c, false);
            }

            for k in 0..channels {
                to.push(<f32 as FromSample<T>>::from_sample(
                    from.chan(k)[ticker.int() as usize],
                ));
            }

            *ticker += dt;
        }

        ((to.len() - og_len) / channels, true)
    }

    /// Returns `false` if at end of stream.
    pub fn take_one(&mut self, dt: U32d32, out: &mut [f32]) -> bool {
        macro_rules! each {
            ($($ty:ident)*) => {
                match self.decoder.last_decoded() {
                    $(AudioBufferRef::$ty(b) => Self::take_one_ref(&mut self.ticker, &*b, dt, out),)*
                }
            };
        }
        while !each!(U8 U16 U24 U32 S8 S16 S24 S32 F32 F64) {
            let packet = match self.format.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == io::ErrorKind::UnexpectedEof =>
                {
                    return false
                }
                Err(e) => todo!("{e}"),
            };
            self.decoder.decode(&packet).unwrap();
        }
        true
    }

    /// Returns `false` if out of frames.
    /// Rewinds ticker when out of frames.
    fn take_one_ref<T: SymSample>(
        ticker: &mut U32d32,
        from: &AudioBuffer<T>,
        dt: U32d32,
        out: &mut [f32],
    ) -> bool
    where
        f32: FromSample<T>,
    {
        let dt = dt * from.spec().rate;
        let channels = from.spec().channels.count();

        assert_eq!(out.len(), channels, "channel count mismatch");

        if ticker.int() as usize >= from.frames() {
            *ticker -= U32d32::from(from.frames() as u32);
            return false;
        }

        for (i, w) in out.iter_mut().enumerate() {
            *w = <f32 as FromSample<T>>::from_sample(from.chan(i)[ticker.int() as usize]);
        }

        *ticker += dt;

        true
    }
}

// Rust is being retarded
struct AsRefImpl(Arc<Box<[u8]>>);

impl AsRef<[u8]> for AsRefImpl {
    fn as_ref(&self) -> &[u8] {
        &**self.0
    }
}

fn make_source(data: Arc<Box<[u8]>>) -> MediaSourceStream {
    let data = Box::new(io::Cursor::new(AsRefImpl(data))) as Box<dyn MediaSource>;
    MediaSourceStream::new(data, MediaSourceStreamOptions::default())
}

#[cfg(test)]
mod test {
    use super::*;
    use util::soa::{Vec2, Vec3};

    /*
    #[test]
    fn play() {
        let out = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("/tmp/test.raw")
            .unwrap();
        let mut out = std::io::BufWriter::new(out);

        //let mut s = Dev::new((1 << 16).try_into().unwrap());
        let mut s = Dev::new((1 << 20).try_into().unwrap());
        dbg!(s.sample_rate(), s.channels());
        s.play();

        let mut t = U32d32::ZERO;
        let mut t2 = U32d32::ZERO;
        //let mut stream = mix::wave::RandomStep::new(rand::rngs::ThreadRng::default(), 1e1);
        let mut stream = mix::wave::RandomStep2::new(rand::rngs::ThreadRng::default(), 1e3);
        while false {
            let v = mix::wave::sine(t.frac()) * stream.current(0);

            //let v = v * mix::transform::lerp(1.0, stream.current(0), 0.9);

            let v =
                mix::transform::square_dropoff(v, 20.0 * (1.0 / 20.0 + 1.0 - t2.frac().to_f32()));

            while s.feed(&[&[v], &[v]]) == 0 {
                std::thread::sleep_ms(10);
            }
            t += s.frame_time() * (U32d32::ONE * 2 - U32d32::from(t2.frac())) * 440;
            t2 += s.frame_time();
            stream.next_sample(s.frame_time());
        }

        //let mut stream_a = mix::Periodic::new(|t| mix::wave::sine(t * 440));
        let mut stream_a = mix::wave::RandomStep2::new(rand::rngs::ThreadRng::default(), 1e3);
        let mut stream_b =
            AudioData::new(Arc::new(include_bytes!("/tmp/test.ogg").to_vec().into()));
        //let mut stream = mix::Combine::new(stream_a, stream_b, |x, y| mix::transform::lerp(x, y, 0.6));
        //let mut stream = mix::Combine::new(stream_a, stream_b, |x, y| x * 2.0 + y);

        //let mut stream = mix::transform::Damp::new(stream_b, 1e-9);
        //let mut stream = mix::Periodic::new(|t| mix::wave::sine(t * 100).powi(1023));
        //let mut stream = mix::transform::Damp::new(stream, 0.999);

        let mut stream = mix::Periodic::new(|t| {
            //(U0d32::MAX - t).to_f32().powi(4) * mix::wave::sine(t * 440)
            if t < U0d32::HALF {
                let t = t * 2;
                let a = 220;
                //let b = U0d32::MAX - (t / 2);
                let b = U0d32::MAX - t;
                //mix::wave::sine(a * b * t)// * (U0d32::MAX - t).to_f32()
                mix::wave::sine(220 * (U0d32::MAX - t).powi(2))
            } else {
                0.0
            }
        });
        let mut stream = AudioData::new(Arc::new(include_bytes!("/tmp/test.ogg").to_vec().into()));
        let mut stream = AudioData::new(Arc::new(include_bytes!("/tmp/test.wav").to_vec().into()));
        //let mut stream = mix::Combine::new(stream, mix::wave::RandomStep2::new(rand::thread_rng(), 1e5), |x, y| x * (1.0 + y));

        //let mut stream = mix::Periodic::new(|t| mix::wave::sine(t.wrapping_mul_u32(200)) * 0.2 + mix::wave::sine(t.wrapping_mul_u32(100)) * 0.8);

        let mut stream = mix::Periodic::new(mix::wave::sine);
        let mut stream = mix::envelope::Linear::new(
            //U32d32::ONE * 208,
            U32d32::ONE * 400,
            1.0,
            Vec3::from([
                /*
                (U0d32::HALF / 12, U32d32::ONE * 800),
                (U0d32::HALF / 10, U32d32::ONE * 624),
                (U0d32::HALF / 8, U32d32::ONE * 400),
                (U0d32::HALF / 6, U32d32::ONE * 400),
                (U0d32::HALF / 2 + U0d32::EPSILON, U32d32::ZERO),
                (U0d32::MAX, U32d32::ZERO),
                */
                (U0d32::HALF, U32d32::ZERO, 1.0),
                (U0d32::MAX, U32d32::ZERO, 1.0),
            ]),
            stream,
        );
        /*
        let mut stream = mix::Combine::new(stream,
            mix::wave::RandomStep2::new(rand::thread_rng(), 1e3), |x, y| mix::lerp(x, y, 1.0));
        */
        let a = mix::Periodic::new(|t| mix::wave::sine(t.wrapping_mul_u32(300)));
        let b = mix::Adjust::new(mix::Periodic::new(mix::wave::sine), |t| t * 3, |s| s);
        let a = mix::Combine::new(a, b, |x, y| mix::lerp(x, x * y, 0.3));

        //let mut stream = mix::Combine::new(a, stream, |x, y| x * y);

        let _ = 4 - core::mem::size_of_val(&mix::Periodic::new(mix::wave::sine));

        //let mut stream = mix::filter::FiniteImpulseResponseSource::new([1.0; 256], stream);
        //let mut stream = mix::transform::Damp::new(stream, 0.99);

        let mut stream = mix::Adjust::new(
            mix::Periodic::new(|t| mix::wave::emilio_pisanty(t, 1e1, 100.0)),
            |t| t * 1,
            |s| s,
        );
        let noise = mix::wave::RandomStep2::new(rand::thread_rng(), 1e3);
        let noise = mix::wave::FullRandom::new(rand::thread_rng());
        let mut stream = mix::Combine::new(stream, noise, |x, y| mix::lerp(x, x * y, 0.3));
        //let mut stream = mix::filter::FiniteImpulseResponseSource::new([1.0; 512], stream);

        let mut stream = mix::Adjust::new(
            //mix::Periodic::new(|t| 1.0 - (mix::wave::sawtooth(U0d32::MAX - t) * 0.5 + 0.5).powi(2)),
            //mix::Periodic::new(|t| 1.0 - mix::wave::sawtooth(t).powi(2)),
            //mix::Periodic::new(mix::wave::sine),
            //mix::Periodic::new(|t| mix::wave::sine(t).abs()),
            mix::Periodic::new(|t| 1.0 - mix::wave::sawtooth(t).powi(4)),
            |t| t * 100,
            |s| s,
        );
        //let mut stream = mix::transform::RandomTimeScale::new(rand::thread_rng(), stream, U32d32::HALF..=U32d32::HALF * 3);
        //let mut stream = mix::transform::RandomTimeScale::new(rand::thread_rng(), stream, U32d32::ZERO..=U32d32::ONE * 2);


        let mut stream = mix::envelope::Linear::new(
            //U32d32::ONE * 208,
            U32d32::ZERO,
            1.0,
            Vec3::from([
                /*
                (U0d32::HALF / 12, U32d32::ONE * 800),
                (U0d32::HALF / 10, U32d32::ONE * 624),
                (U0d32::HALF / 8, U32d32::ONE * 400),
                (U0d32::HALF / 6, U32d32::ONE * 400),
                (U0d32::HALF / 2 + U0d32::EPSILON, U32d32::ZERO),
                (U0d32::MAX, U32d32::ZERO),
                */
                (U0d32::from_f32(0.1), U32d32::ZERO, 1.0),
                (U0d32::from_f32(0.15), U32d32::ONE * 6, 1.0),
                (U0d32::from_f32(0.3), U32d32::ONE * 6, 1.0),
                (U0d32::from_f32(0.45), U32d32::ONE * 10, 1.0),
                (U0d32::from_f32(0.55), U32d32::ONE * 20, 1.0),
                (U0d32::from_f32(0.8), U32d32::ONE * 30, 1.0),
                (U0d32::from_f32(0.95), U32d32::ONE * 30, 1.0),
                (U0d32::MAX, U32d32::ZERO, 0.0),
            ]),
            stream,
        );
        //let mut stream = mix::filter::FiniteImpulseResponseSource::new(stream, [1.0; 512], U32d32::ONE / 1);
        //let mut stream = mix::filter::FiniteImpulseResponseSource::new(stream, [1.0; 512], U32d32::ONE / 384000);
        let mut stream = mix::Adjust::new(stream, |t| t / 10, |s| s);

        let stream = mix::Combine::new(
            mix::Periodic::new(|t| mix::neg_exp(U32d32::from(t) * 50 / 5)),
            mix::Adjust::new(mix::Periodic::new(mix::wave::sine), |t| t * 200 / 5, |s| s),
            //mix::wave::FullRandom::new(rand::thread_rng()),
            //mix::wave::RandomStep2::new(rand::thread_rng(), 1e3),
            //mix::wave::RandomStep::new(rand::thread_rng(), 1e1),
            //mix::transform::RandomTimeScale::new(rand::thread_rng(), mix::Adjust::new(mix::Periodic::new(mix::wave::sine), |t| t * 200 / 5, |s| s), U32d32::HALF..=U32d32::HALF * 3),
            |x, y| x * y,
        );
        let stream = mix::Adjust::new(stream, |t| t * 5, |s| s);

        let noise = mix::wave::FullRandom::new(rand::thread_rng());
        //let mut stream = mix::Combine::new(stream, noise, |x, y| mix::lerp(x, x * y, 5e-3));



        let points = (1..=64)
            .map(|i| U0d32::MAX / 64 * i)
            .map(|i| (i, U32d32::from(i) * 8))
            .map(|(t, u)| (t, U32d32::from_f32((-u.to_f32() * 3.0).exp()), (-u.to_f32()).exp()))
            .collect::<Vec3<_, _, _>>();
        let stream = mix::envelope::Linear::new(
            U32d32::ONE,
            1.0,
            points,
            mix::wave::FullRandom::new(rand::thread_rng()),
            //mix::Adjust::new(mix::Periodic::new(mix::wave::sine), |t| t * 10000, |s| s),
        );
        /*
        let mut stream = mix::Adjust::new(
            mix::Periodic::new(|t| mix::wave::emilio_pisanty(t, 1e1, 100.0)),
            |t| t * 1,
            |s| s,
        );
        */
        let mut stream = mix::limit_duration(stream, U32d32::ONE);
        //let mut stream = mix::Adjust::new(stream, |t| t / 10, |s| s);

        while stream.next_sample(s.frame_time()) {
            while s.feed(&[&[stream.current(0)], &[stream.current(1)]]) == 0 {
                std::thread::sleep_ms(10);
            }
            std::io::Write::write_all(&mut out, &stream.current(0).to_le_bytes()).unwrap();
            //stream.set_acceleration(1.0 + mix::wave::sine(t.frac().to_f32()).abs());
            t += s.frame_time();
        }
        while !s.is_empty() {
            std::thread::sleep_ms(10);
        }
        std::thread::sleep_ms(100);
    }
    */

    #[test]
    fn play() {
        let wait = std::sync::Arc::new(std::sync::Condvar::new());
        let wait2 = wait.clone();

        let mut stream = AudioData::new(Arc::new(include_bytes!("/tmp/test.ogg").to_vec().into()));

        let f = move |buf: &mut [f32], channels: u16, dt: U0d32| {
            let mut it = buf.chunks_mut(usize::from(channels));
            for b in &mut it {
                if !stream.take_one(dt.into(), b) {
                    it.for_each(|b| b.fill(0.0));
                    wait2.notify_all();
                    break;
                }
            }
        };

        let mut s = Dev::new(
            &DevConfig {
                //buffer_size: BufferSize::MaxLatency(U0d32::MAX / 200), // 5ms
                buffer_size: BufferSize::MaxLatency(U0d32::MAX / 100), // 10ms
                channels: 2,
                //sample_rate: 48000,
                sample_rate: 48000,
            },
            f,
        );
        s.play();

        let stub = std::sync::Mutex::new(());
        drop(wait.wait(stub.lock().unwrap()).unwrap());

        s.pause();

        /*

        let mut buf = Vec::new();
        while stream.next_samples(s.frame_time(), &mut buf, 1) {
            let mut n = 0;
            while n < buf.len() {
                n += s.feed_interleaved(&buf[n..]);
                while s.is_full() {
                    std::thread::sleep_ms(10);
                }
            }
            buf.clear();
        }
        while !s.is_empty() {
            std::thread::sleep_ms(10);
        }
        std::thread::sleep_ms(100);
        */
    }
}
