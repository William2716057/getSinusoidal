import wave
import numpy as np
import matplotlib.pyplot as plt

def readWav(filename):
	with wave.open(filename, 'rb') as wav_file:
		n_channels = wav_file.getnchannels()
		sample_width = wav_file.getsampwidth()
		framerate = wav_file.getframerate()
		n_frames = wav_file.getnframes()
		frames = wav_file.readframes(n_frames)
		if sample_width == 1:
			dtype = np.uint8
		elif sample_width == 2:
			dtype = np.int16
		elif sample_width == 4:
			dtype = np.int32
		else:
			raise ValueError("Unsupported sample width")
		audio_data = np.frombuffer(frames, dtype=dtype)
		audio_data = audio_data.astype(np.float32)
		audio_data /= np.iinfo(dtype).max
		return audio_data, framerate

def getSinusoidalValues(audio_data, framerate, frequency_range=(20, 20000)):
	#Calculate time axis
	time_axis = np.arange(len(audio_data)) / framerate
    
	#Perform Fast Fourier Transform (FFT) to get frequency domain information
	fft_result = np.fft.fft(audio_data)
	frequencies = np.fft.fftfreq(len(fft_result), d=1/framerate)
    
	#Filter frequencies within the specified range
	mask = np.logical_and(frequencies >= frequency_range[0], frequencies <= frequency_range[1])
	filtered_fft = fft_result[mask]
	filtered_frequencies = frequencies[mask]
    
	#Convert back to time domain using inverse FFT
	filtered_audio = np.fft.ifft(filtered_fft)
    
	return time_axis, np.real(filtered_audio)


filename = 'wave.wav'
audioData, framerate = readWav(filename)
timeAxis, sinusoidalValues = getSinusoidalValues(audioData, framerate)

# Plot sinusoidal values
plt.figure(figsize=(10, 5))
plt.plot(timeAxis, sinusoidalValues)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sinusoidal Values ')
plt.grid(True)
plt.show()
