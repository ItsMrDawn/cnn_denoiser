import tensorflow as tf
import numpy as np
import librosa
import os

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(noisy_mag_segments, clean_magnitude, noisy_phase):
    feature = {
        'noisy_mag_segments': _bytes_feature(noisy_mag_segments.tobytes()),
        'clean_magnitude': _bytes_feature(clean_magnitude.tobytes()),
        'noisy_phase': _bytes_feature(noisy_phase.tobytes())
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def preprocess(audio, sr, n_fft=256, hop_length=64, fixed_length=256, window='hamming'):
    # Compute the STFT with windowing
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    magnitude, phase = librosa.magphase(stft)
    phase_angle = np.angle(phase)  # Extract the angle of the complex numbers
    
    # Pad or truncate the magnitude and phase to the fixed length
    if magnitude.shape[1] < fixed_length:
        pad_width = fixed_length - magnitude.shape[1]
        magnitude = np.pad(magnitude, ((0, 0), (0, pad_width)), mode='constant')
        phase_angle = np.pad(phase_angle, ((0, 0), (0, pad_width)), mode='constant')
    else:
        magnitude = magnitude[:, :fixed_length]
        phase_angle = phase_angle[:, :fixed_length]
    
    return magnitude, phase_angle

def prepare_input_features(stft_features, num_segments, num_features):
    noisy_stft = np.concatenate([stft_features[:, 0:num_segments - 1], stft_features], axis=1)
    stft_segments = np.zeros((num_features, num_segments, noisy_stft.shape[1] - num_segments + 1))
    for index in range(noisy_stft.shape[1] - num_segments + 1):
        stft_segments[:, :, index] = noisy_stft[:, index:index + num_segments]
    return stft_segments


def save_to_tfrecord(clean_folder_path, noisy_folder_path, output_tfrecord, num_segments=8, num_features=129, fixed_length=256):
    clean_files = sorted([f for f in os.listdir(clean_folder_path) if f.endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(noisy_folder_path) if f.endswith('.wav')])
    i = 0
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for clean_filename in clean_files:
            clean_file_path = os.path.join(clean_folder_path, clean_filename)
            clean_audio, sr = librosa.load(clean_file_path, sr=16000)
            clean_magnitude, _ = preprocess(clean_audio, sr, fixed_length=fixed_length)
            
            # Find corresponding noisy samples
            corresponding_noisy_files = [f for f in noisy_files if clean_filename in f]
        
            i += 1

            if i == 50:
                break

            for noisy_filename in corresponding_noisy_files:
                noisy_file_path = os.path.join(noisy_folder_path, noisy_filename)
                noisy_audio, sr = librosa.load(noisy_file_path, sr=16000)
                noisy_magnitude, noisy_phase = preprocess(noisy_audio, sr, fixed_length=fixed_length)
                
                # Segment the noisy magnitude
                noisy_segments = prepare_input_features(noisy_magnitude, num_segments, num_features)

                print(f'Clean magnitude: {clean_magnitude.shape}')
                print(f'Noisy segments: {noisy_segments.shape}')
                print(f'Noisy phase: {noisy_phase.shape}')
                
                # Serialize the data
                serialized_example = serialize_example(noisy_segments, clean_magnitude, noisy_phase)
                
                # Write the serialized data to the TFRecord file
                writer.write(serialized_example)

# Example usage
clean_folder_path = 'dataset/CleanSpeech_training'
noisy_folder_path = 'dataset/NoisySpeech_training'
output_tfrecord = 'speech_denoising.tfrecord'
save_to_tfrecord(clean_folder_path, noisy_folder_path, output_tfrecord)

# todo: add the code to read the tfrecord file and decode the features