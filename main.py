
import torch, soundfile
from model import SE
from tools import *

name = '001'
input_video_name = 'data/%s.mp4'%name
out_video_name   = 'exps/%s_res.mp4'%name
out_audio_name   = 'exps/%s_res.wav'%name
mix_audio_name   = 'exps/%s.wav'%name

# Step 1: Load the audio
mix_audio = load_audio(input_video_name, mix_audio_name)
# Step 2: Load the visual
faces     = load_visual(input_video_name)

# Step 3: Preprocess, align the length
mix_audio, faces = preprocess(mix_audio, faces)
model = SE().cuda()
#print(model)
# Step 4: Evaluate and save the results
with torch.no_grad():
    out = model(mix_audio, faces)[0]
print('finished')
# Step 5: Save the results
save_results(input_video_name, out_audio_name, out_video_name, out)