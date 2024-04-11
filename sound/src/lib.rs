use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SampleFormat, SizedSample,
};

pub struct Sound {
    host: cpal::Host,
    device: cpal::Device,
    stream: cpal::Stream,
}

impl Sound {
    pub fn new() -> Self {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .expect("no output device available");

        let mut supported_configs_range = device
            .supported_output_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_max_sample_rate();

        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
        let sample_format = supported_config.sample_format();
        let config = supported_config.config();

        // Produce a sinusoid of maximum amplitude.
        let sample_rate = config.sample_rate.0 as f32;
        let channels = config.channels as usize;
        let mut sample_clock = 0f32;
        let mut next_value = move || {
            sample_clock = (sample_clock + 1.0) % sample_rate;
            (sample_clock * 440.0 * 2.0 * std::f32::consts::PI / sample_rate).sin()
        };

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [u8], _: &cpal::OutputCallbackInfo| {
                    write_data(data, channels, &mut next_value)
                },
                err_fn,
                None,
            )
            .unwrap();

        Self {
            host,
            device,
            stream,
        }
    }
}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let value: T = T::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn play() {
        let s = Sound::new();

        s.stream.play().unwrap();

        std::thread::sleep_ms(2000);

        s.stream.pause().unwrap();
    }
}
