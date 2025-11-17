import soundfile, cv2, subprocess, numpy, torch, ffmpeg

def load_audio(input_video_name, mix_audio_name):
	command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads 10 -ar 16000 %s -loglevel panic" % (input_video_name, mix_audio_name))
	subprocess.call(command, shell=True, stdout=None)
	audio, _ = soundfile.read(mix_audio_name)
	return audio

def load_visual(input_video_name): 	
	V = cv2.VideoCapture(input_video_name)
	faces = []
	while(V.isOpened()):	
		ret, face = V.read()
		if ret == True:
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			face = cv2.resize(face, (224,224))
			face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
			faces.append(face)
		else:
			break
	return numpy.array(faces)

def preprocess(mix_audio, faces):
	audio_length = faces.shape[0] * 640
	if mix_audio.shape[0] < audio_length:
		shortage  = audio_length - audio.shape[0]
		audio_length     = numpy.pad(audio_length, (0, shortage), 'wrap')
	mix_audio = mix_audio[:audio_length]

	mix_audio = torch.FloatTensor(mix_audio).cuda().unsqueeze(0)
	faces = torch.FloatTensor(faces).cuda().unsqueeze(0)

	return mix_audio, faces

def save_results(input_video_name, out_audio_name, out_video_name, out):
	# Save the audio
	out = out.cpu().numpy()
	soundfile.write(out_audio_name, out, 16000)
	# Save the video extract the visual feature from original video and attach the audio feature after filter to the visual feature
	output_stream = ffmpeg.output(ffmpeg.input(input_video_name).video, ffmpeg.input(out_audio_name), out_video_name, vcodec='copy', acodec='aac', strict='experimental', loglevel='error')
	ffmpeg.run(output_stream, overwrite_output=True)
