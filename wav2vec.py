import torch
import torchaudio

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.__version__)
print(torchaudio.__version__)
print(device)

bundle = torchaudio.pipelines.WAV2VEC2_BASE

print("Sample Rate:", bundle.sample_rate)

model = bundle.get_model().to(device)

print(model.__class__)

import os

input_training_file = 'LJSpeech-1.1/training.txt'
input_wavs_dir = 'LJSpeech-1.1/wavs'
with open(input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

len(training_files)

output_dir = 'wav2vec'

for file in training_files:
    waveform, sample_rate = torchaudio.load(file)
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    # with torch.inference_mode():
    #     features, _ = model.extract_features(waveform)

    res, _ = model(waveform)

    output_file = os.path.join(output_dir, file.split('/')[-1].split('.')[0] + '.pt')

    torch.save(res, output_file)
    print(res.size())
