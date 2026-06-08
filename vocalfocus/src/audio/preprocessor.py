from . import audio_utils
from . import audio_config as au
import io
import wave
import librosa
import numpy as np

logger = audio_utils.setup_logger(__name__,10)



def load_waveform(path_or_bytes):

    logger.info("Start load_waveform")

    #step 0 check if file(.wav) or buffer(stream)

    if isinstance(path_or_bytes, bytes):
        logger.info("Input is bytes from buffer")

        buffer = io.BytesIO(path_or_bytes)
        sr = au.SAMPLE_RATE
        waveform = audio_utils.bytes_to_pcm_array(buffer.read())
        #
    elif isinstance(path_or_bytes, str):
        logger.info("Input is a file")

        with wave.open(path_or_bytes, 'rb') as wf:
            sr = wf.getframerate()
            raw_bytes = wf.readframes(wf.getnframes())
        waveform = audio_utils.bytes_to_pcm_array(raw_bytes)

        #
    else:
        raise TypeError("input is not from buffer nor file")

    return waveform, sr

    
def resample(waveform, orig_sr, target_sr=au.SAMPLE_RATE):

    logger.info("Start resample")

    # set things up
    waveform = np.asarray(waveform, dtype=np.float32)

    # if both sr are same
    if orig_sr == target_sr:
        logger.info("Have the same Sample rate")
        return waveform

    # not same
    logger.info("Not the same Sample rate")
    waveform_resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_fast")
    waveform_resampled = np.clip(waveform_resampled, -1.0, 1.0)
    waveform_resampled = np.asarray(waveform_resampled, dtype=np.float32)

    return waveform_resampled


def vad_trim(waveform, thresh_db=-40, eps=1e-12):

    logger.info("Start vad_trim")

    #set things up
    wave_array = np.asarray(waveform, dtype=np.float32)
    frames = audio_utils.frame_audio(wave_array)
    n_frames = frames.shape[0]
    hop = au.HOP_LENGTH
    win = au.WIN_LENGTH

    # calculate RMS for each frame
    rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
    rms = np.asarray(rms, dtype=np.float32)
    db = 20.0 * np.log10(rms + eps)
    
    # find indices of frames
    speech_idx = np.where(db > thresh_db)[0]

    # if silent frame detected return 
    if speech_idx.size == 0:
        logger.info("No speech detected above threshold; returning waveform unchanged")
        return wave_array

    # Conservative frames before and after detected speech(rule no char loss) 
    margin = 2
    start_frame = max(0, int(speech_idx[0]) - margin)
    end_frame   = min(n_frames - 1, int(speech_idx[-1]) + margin)

     
    start_sample = start_frame * hop
    end_sample   = end_frame * hop + win  
    trimmed = wave_array[start_sample:end_sample]

    logger.debug(f"Trimmed samples: start={start_sample}, end={end_sample}, length={trimmed.shape[0]}")
    return trimmed.astype(np.float32)


# coef = pre-emphasis coefficient
def pre_emphasis(waveform, coef=0.97):

    logger.info("Start pre_emphasis")

    wave_array = np.asarray(waveform, dtype=np.float32)

    if wave_array.size == 0:
        logger.warning("pre_emphasis received empty waveform")
        return wave_array

    # empty copy of wave_array
    emphasized = np.empty_like(wave_array, dtype=np.float32)

    # for first sample(common convention)
    emphasized[0] = wave_array[0]

    # vectorized application for the rest
    if wave_array.shape[0] > 1:
        emphasized[1:] = wave_array[1:] - coef * wave_array[:-1]

    logger.debug(
        "pre_emphasis applied: coef=%s, input_peak=%.6f, output_peak=%.6f",
        coef, float(np.max(np.abs(wave_array))), float(np.max(np.abs(emphasized)))
    )

    return emphasized.astype(np.float32)

def pre_emphasis_stream(waveform, coef=0.97, prev_sample=None):

    logger.info("Start pre_emphasis_stream")

    # ensure numpy array and dtype
    wave_array = np.asarray(waveform, dtype=np.float32)

    # handle empty input
    if wave_array.size == 0:
        logger.warning("pre_emphasis_stream received empty waveform")
        return wave_array, prev_sample

    # allocate output
    emphasized = np.empty_like(wave_array, dtype=np.float32)

    # first sample: use prev_sample if provided, otherwise common convention
    if prev_sample is None:
        emphasized[0] = wave_array[0]
    else:
        emphasized[0] = wave_array[0] - coef * np.float32(prev_sample)

    # vectorized application for the rest of the chunk
    if wave_array.shape[0] > 1:
        emphasized[1:] = wave_array[1:] - coef * wave_array[:-1]

    # compute last sample to return for next chunk continuity
    last_sample = float(wave_array[-1])

    logger.debug(
        "pre_emphasis_stream applied: coef=%s, input_peak=%.6f, output_peak=%.6f, prev_sample=%s",
        coef, float(np.max(np.abs(wave_array))), float(np.max(np.abs(emphasized))), str(prev_sample)
    )

    return emphasized.astype(np.float32), last_sample





