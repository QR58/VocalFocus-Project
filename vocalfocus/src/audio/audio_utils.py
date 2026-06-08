from . import audio_config
import logging
import numpy as np
import librosa


# inital logging


def setup_logger(name, level):
    
    #creat logger with the file(source) name with level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # creat handler
    stream_handler = logging.StreamHandler()

    #set the print format: level | file_name | message | time
    formatter = logging.Formatter('%(levelname)s | %(name)s | %(message)s | %(asctime)s')

    #connect everything
    stream_handler.setFormatter(formatter)
    #check if handler exisit
    if not logger.hasHandlers():
       logger.addHandler(stream_handler)

    
    return logger

logger = setup_logger(__name__,10)
# convert raw bytes to PCM array

def bytes_to_pcm_array(raw_bytes):


   logger.info("Start converting from bytes to PCM")
   
   # interpret bytes as int16 array
      #check if data is empty ,odd or even
   if len(raw_bytes) == 0:
      logger.warning("Empty data")
      return np.zeros(0, dtype=np.float32)
      
   if len(raw_bytes) % 2 != 0:
         #odd
         logger.warning("Odd-length buffer")
         #trim the last byte
         raw_bytes = raw_bytes[:-1]
         logger.info("Trimmed the last byte to make length even")
         

   #even 
   array = np.frombuffer(raw_bytes, dtype=np.int16)
   logger.debug(f"Number of Samples = {len(array)}")

   # convert array to float32 and normalize (-1,+1)
   float32_array = array.astype(np.float32) / 32768.0

   # resharp channel type 
   if audio_config.CHANNELS > 1:
         if len(float32_array) % audio_config.CHANNELS != 0:
            logger.warning("Sample count not divisible by CHANNELS; truncating extra samples")
            trim = len(float32_array) - (len(float32_array) // audio_config.CHANNELS) * audio_config.CHANNELS
            float32_array = float32_array[:-trim]
         
         float32_array = float32_array.reshape(-1, audio_config.CHANNELS)
         logger.debug(f"Reshaped to {float32_array.shape}")
         

   return float32_array

def convert_to_array(raw_bytes: bytes) -> np.ndarray:
    """
    Convert raw interleaved PCM bytes from the microphone into a mono
    float32 NumPy array in the range [-1.0, 1.0].

    This is the helper expected by AudioPipeline._process_frame().
    """
    # Existing helper: returns a NumPy array from raw bytes
    pcm = bytes_to_pcm_array(raw_bytes)

    # If audio is multi-channel (shape: [samples, channels]), downmix to mono
    if pcm.ndim == 2 and pcm.shape[1] > 1:
        pcm = pcm.mean(axis=1)

    # Ensure float32 dtype for the pipeline
    return pcm.astype(np.float32, copy=False)
   
# convert pcm array to raw data
def pcm_array_to_bytes(input_array):

   logger.info("Start converting from PCM to bytes")

   # set the array as numpy array
   input_array = np.asarray(input_array)

   # checking if array is empty
   if len(input_array) == 0:
      logger.warning("Empty array")
      return b""
   
   # ensure array is in float32(-1,+1)
   if input_array.dtype != np.float32:
      original_dtype = input_array.dtype
      input_array  = input_array.astype(np.float32)
      logger.debug(f" array cast from {original_dtype} to float32")
   
   # clip values to (-1, +1)
   input_array = np.clip(input_array, -1.0, 1.0)
   logger.debug(f"values clipped, min={input_array.min()}, max={input_array.max()}")


   # scale float32 values to int16 range
 
   scaled_array = input_array * 32768.0 
   int16_array = scaled_array.astype(np.int16)
   logger.debug(f"array scaled to int16, shape={int16_array.shape}")
   logger.debug(f"min={int16_array.min()}, max={int16_array.max()}")
   
   # Convert to raw bytes
   raw_bytes = int16_array.tobytes()
   logger.debug(f"converted {len(int16_array)} samples into {len(raw_bytes)} bytes")

    # Step 7: Return result
   return raw_bytes

# Channel Handling 
def reshape_channels(input_array):
   
   logger.info("Start reshape array channels")
   # set the array as numpy array
   input_array = np.asarray(input_array)

   logger.debug(f"The original shape:{input_array.shape}, and the data type:{input_array.dtype}")
   
   # Validate dimensions
   if input_array.ndim == 1:
      logger.info("Audio is mono, it is 1D array")
      detected_channels = 1
   elif input_array.ndim == 2:
      logger.info("Audio is stereo, it is 2D array")
      
      x,y = input_array.shape
      #check channel position(first,last)?
      if y <= 8 and x > y:
         #channel last
         input_array = input_array.T
      detected_channels = input_array.shape[0]
      logger.debug(f"forced to channels-first, shape={input_array.T.shape}, channels={detected_channels}")
   else:
      raise ValueError("Only 1D AND 2D Audio are supported")
   
   # ensure type float32
   if input_array.dtype != np.float32:
      input_array = input_array.astype(dtype=np.float32)

   # force to mono

   if detected_channels > 1:
    input_array = input_array.mean(axis=0)
    logger.info("Downmixed to mono using mean across channels")

   logger.debug(f"final-shape={input_array.shape} , data-type={input_array.dtype}")

   return input_array


def normalize_audio(input_array, target_level=-20.0, eps=1e-12, max_gain_db=2000.0):
    """
    Normalize a 1-D mono signal to a target RMS level in dBFS,
    but clamp the maximum applied gain to avoid crazy amplification
    on near-silent segments.

    target_level: desired RMS level in dBFS (e.g., -20 dBFS).
    max_gain_db: maximum gain (in dB) that will be applied.
    """
    logger.info("Start normalize_audio")
    x = np.asarray(input_array, dtype=np.float32)

    if x.ndim != 1:
        raise ValueError("normalize_audio expects 1-D mono array")

    # RMS + current level in dB
    rms = np.sqrt(np.mean(np.square(x)))
    current_db = 20.0 * np.log10(rms + eps)

    # If truly near-silence, skip normalization
    if rms <= eps or current_db < -80.0:
        logger.warning(
            "Near-silence detected (RMS≈0 or < -80dB); skipping gain change"
        )
        return x

    # Desired gain in dB
    gain_db = target_level - current_db

    # Clamp gain to avoid wild boosts
    if gain_db > max_gain_db:
        logger.debug(
            "Requested gain_db=%.2f > max_gain_db=%.2f; clamping.",
            gain_db,
            max_gain_db,
        )
        gain_db = max_gain_db
    elif gain_db < -max_gain_db:
        logger.debug(
            "Requested gain_db=%.2f < -max_gain_db=%.2f; clamping.",
            gain_db,
            -max_gain_db,
        )
        gain_db = -max_gain_db

    gain_linear = 10.0 ** (gain_db / 20.0)
    logger.debug("Using gain_db=%.2f, gain_linear=%.4f", gain_db, gain_linear)

    y = x * gain_linear
    y = np.asarray(y, dtype=np.float32)
    post_peak = float(np.max(np.abs(y)))

    # Final safety clipping if we exceeded [-1, 1]
    if post_peak > 1.0:
        reduce_gain = 1.0 / post_peak
        logger.warning(
            "normalize_audio peak %.4f > 1.0; scaling down by %.4f",
            post_peak,
            reduce_gain,
        )
        y *= reduce_gain

    return y.astype(np.float32)
 

def frame_audio(input_array):

   logger.info("Start frame audio")
   # step 0 set the parameters
   input_array = np.asarray(input_array)
   frame_size = audio_config.WIN_LENGTH
   hop_size = audio_config.HOP_LENGTH

   # samples in array
   n = len(input_array)

   # empty array
   if n == 0:
      logger.warning("Empty array")
      return np.zeros((0,frame_size), dtype=input_array.dtype)

   #cal frames number and pad
   n_frames = max(1, int(np.ceil((n - frame_size) / hop_size)) + 1)
   total_needed = ((n_frames - 1) * hop_size) + frame_size
   pad_amount = max(0, total_needed - n)

   if pad_amount > 0:
      logger.debug(f"Padding with {pad_amount} zeros")
      input_array = np.concatenate([input_array, np.zeros(pad_amount, dtype=input_array.dtype)])
   
   # create frames array

   frames = np.zeros((n_frames, frame_size), dtype=input_array.dtype)

   for i in range(n_frames):
      start = i * hop_size
      end = start + frame_size
      frames[i, :] = input_array[start:end]
   
   logger.info(f"Framed audio into {n_frames} frames of size {frame_size}")
   return frames


def apply_window(frame):
   
   logger.info("Start apply window")
   #ensure it is np array
   frames = np.asarray(frame)
   # check if it is 1D or 2D 
   if frames.ndim == 1:
        n = frames.shape[0]   
   elif frames.ndim == 2:
        n = frames.shape[1]  
   else:
        raise ValueError("apply window expects 1D or 2D array")
   
   window = np.hanning(n)
   logger.debug(f"Window shape={window.shape}, min={window.min()}, max={window.max()}")
   
   # apply the window
   if frames.ndim == 1:
        windowed_frame = frames * window
   else:  # 2D
        windowed_frame = frames * window[np.newaxis, :]

   logger.debug(f"Applied window, frame min={windowed_frame.min()}, max={windowed_frame.max()}")

   return windowed_frame
   
def stft(input_array):

   logger.info("Start STFT")

   # set array to np array and parameters
   input_array = np.asarray(input_array, dtype=np.float32)
   win_length = audio_config.WIN_LENGTH
   n_fft = audio_config.N_FFT

   # step 1 check 1D or 2D array
   if input_array.ndim == 1:
      frame_length = input_array.shape[0]
      if frame_length != win_length:
         raise ValueError(f"STFT expected 1D frame of win_length={win_length} but got {frame_length}")
      frames_2D = input_array.reshape(1, win_length)

   elif input_array.ndim == 2:
      n_frames, frame_length = input_array.shape
      if frame_length != win_length:
         raise ValueError(f"STFT expected 2D frames with win_length={win_length} but got {frame_length}")
      frames_2D = input_array

   else:
      raise ValueError("STFT expects 1D or 2D array")

   n_frames = frames_2D.shape[0]
   frame_length = frames_2D.shape[1]

   #step 2 Validate N_FFT constraint and compute bins
   if n_fft >= win_length:
      n_bins = (n_fft // 2) + 1
      logger.debug(f"n_bins = {n_bins}")
   else:
      raise ValueError(f"STFT expected n_fft >= win_length ({win_length}), but got {n_fft}")
   
   #step 3 Zero-pad frames to N_FFT
   padded_frames = np.zeros((n_frames, n_fft), dtype=np.float32)

   for i in range(n_frames):
      padded_frames[i, :win_length] = frames_2D[i]
   logger.debug(f"Padded frames shape={padded_frames.shape}, dtype={padded_frames.dtype}")

   #step 4 Compute FFT per frame
   stft_matrix = np.zeros((n_frames, n_bins), dtype=np.complex64)

   for i in range(n_frames):
      spectrum = np.fft.rfft(padded_frames[i], n=n_fft)
      stft_matrix[i, :] = spectrum  
   logger.debug(f"Computed STFT shape={stft_matrix.shape}, dtype={stft_matrix.dtype}")

   return stft_matrix

def Istft(stft_matrix):

   logger.info("Start ISFTF")

   # step 0 set constant
   stft_matrix = np.asarray(stft_matrix, dtype=np.complex64)
   win_length = audio_config.WIN_LENGTH
   hop_length = audio_config.HOP_LENGTH
   n_fft = audio_config.N_FFT
   eps=1e-12

   logger.debug(f"input SFTF shape= {stft_matrix.shape}, dtype= {stft_matrix.dtype}")

   #step 1 validate input shape and derive n_bins and n_frames

   if stft_matrix.ndim != 2:
      raise ValueError("ISFTF expects 2-D stft matrix (n_frames x n_bins)")
   
   n_frames, n_bins = stft_matrix.shape

   expected_n_bins = (n_fft // 2) + 1
   if expected_n_bins != n_bins:
      raise ValueError(f"stft matrix n_bins mismatch: expected {expected_n_bins}, but got {n_bins}")

   #step 2 prepare window and output buffer size estimate

   window = np.hanning(win_length).astype(np.float32)
   out_len = ((n_frames - 1) * hop_length) + win_length
   # accumulation buffer
   output = np.zeros(out_len, dtype=np.float32)
   # for overlap-add normalization
   window_sum  = np.zeros(out_len, dtype=np.float32)

   #step 3 iterate frames, irfft and overlap-add
   for i in range(0, n_frames):

      spectrum_pos = stft_matrix[i, :] # length = n_bins, complex
      frame_time_full = np.fft.irfft(spectrum_pos, n=n_fft) # dtype float, length n_fft
      frame_time = frame_time_full[0 : win_length] # length win_length
      windowed_frame = frame_time * window  # element-wise multiply

   
      # compute output write indices
      start = i * hop_length
      end = start + win_length

      output[start:end] += windowed_frame
      window_sum[start:end] += (window * window) 

   # Step 4: normalize overlap-add(avoid amplitude modulation)

   nonzero_mask = window_sum > eps
   output[nonzero_mask] = output[nonzero_mask] / window_sum[nonzero_mask]
   output[~nonzero_mask] = 0.0

   # Step 5: post-processing checks and scaling 
   pre_clip_peak = np.float32(np.max(np.abs(output)))

   if pre_clip_peak > 1.0:
      reduce_gain = 1.0 / pre_clip_peak
      output *= reduce_gain
      logger.warning(f"Peak {pre_clip_peak:.4f} > 1.0, scaled down by {reduce_gain:.4f}")

   return output
  
def mel_spectrogram(input_array):

   logger.info("Start mel_spectrogram")
   
   #step 0 set data
   input_array = np.asarray(input_array,dtype=np.float32)
   sample_rate = audio_config.SAMPLE_RATE
   n_fft = audio_config.N_FFT
   n_mels = audio_config.N_MELS
   eps = 1e-12

   #step 1 preconditions and normalization advice
   if input_array.ndim != 1:
      raise ValueError("mel_spectrogram expects 1-D mono array")

   #step 2 compute STFT magnitudes
   frames = frame_audio(input_array)
   windowed_frames = apply_window(frames)
   stft_matrix = stft(windowed_frames)
   magnitude = np.abs(stft_matrix)
   power_spec = np.square(magnitude)

   #step 3 build mel filterbank
   mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=70, fmax=8000)
   mel_fb = np.asarray(mel_fb, dtype=np.float32)
   logger.debug(f"Mel filterbank shape={mel_fb.shape}")

   #step 4 apply filterbank
   mel_spec = np.dot(power_spec, mel_fb.T)
   mel_spec = np.asarray(mel_spec, dtype=np.float32)
   logger.debug(f"Mel spectrogram shape={mel_spec.shape}")

   # step 5 Floor to avoid zeros
   mel_spec = np.maximum(mel_spec, eps)

   return mel_spec

def power_to_db(input_array):

   logger.info("Start power_to_db")

   input_array = np.asarray(input_array,dtype=np.float32)
   db_array = librosa.power_to_db(input_array, ref=1.0, top_db=80)

   return db_array


def db_to_power(input_array):

   logger.info("Start db_to_power")

   input_array = np.asarray(input_array,dtype=np.float32)
   pow_array = librosa.db_to_power(input_array, ref=1.0)

   return pow_array

def pad_or_trim(input_array, target_length=16000, pad_value=0.0):
   
    logger.info("Start pad_or_trim")

    # Validate target_length
    if not isinstance(target_length, int):
        raise ValueError("target_length must be an integer")
    if target_length < 0:
        raise ValueError("target_length must be non-negative")

    # Convert to numpy array (do not copy unnecessarily)
    arr = np.asarray(input_array)
    if arr.ndim != 1:
        raise ValueError("pad_or_trim expects a 1-D array")

    n = arr.shape[0]
    logger.debug(f"pad_or_trim input length={n}, target_length={target_length}")

    # Fast path: already correct length
    if n == target_length:
        # ensure float32 and return a copy to avoid accidental in-place edits elsewhere
        if arr.dtype == np.float32:
            logger.debug("Input already correct length and dtype float32; returning copy")
            return arr.copy()
        logger.debug(f"Casting input from {arr.dtype} to float32 and returning copy")
        return arr.astype(np.float32, copy=True)

    # Trim if longer
    if n > target_length:
        logger.debug(f"Trimming from {n} to {target_length} samples")
        return arr[:target_length].astype(np.float32, copy=True)

    # Pad if shorter
    pad_amount = target_length - n
    logger.debug(f"Padding with {pad_amount} samples (pad_value={pad_value})")
    # Ensure base array is float32
    base = arr.astype(np.float32, copy=True)
    if pad_amount <= 0:
        return base
    pad_block = np.full(pad_amount, pad_value, dtype=np.float32)
    result = np.concatenate([base, pad_block])
    logger.debug(f"Padded result length={result.shape[0]}")
    return result

