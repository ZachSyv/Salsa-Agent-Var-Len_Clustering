# Todo: use this in data preprocess cache generation step.



from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


device=torch.device('cpu')
config_path = 'WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'

# config_path = "./configs/xxx.yaml"
# model_path = "./xxx.ckpt"
model_path = 'WavTokenizer/results/train/wavtokenizer_large_unify_600_24k.ckpt'
audio_path = r"S:\Payam\Dance_Salsa_SFU\salsa project\salsa project\Animations\Pair1_8_7_take1_1.wav"
audio_outpath = r"S:\Payam\Dance_Salsa_SFU\salsa project\salsa project\Animations\Pair1_8_7_take1_1_UUUU.wav"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)


wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1)
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
torchaudio.save(audio_outpath, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)