###
# Copyright 2022 Wolfy Fiorini
# All rights reserved
###

import glob
import shutil
import NST
import Process
import os
from PIL import Image
import myconfig as prefs



def rename_files(dir, fr, files, isStyle) :
	temp = []
	print("\n\n", files)
	for i in range(len(files)) :
		file = files[i]
		if isStyle :
			print("Turning ", file, " into ", "Style{}-0".format((i * fr) + 1))
			os.rename(file, dir + "/Style{}-0.jpg".format(i * fr + 1))
			temp.append(dir + "/Style{}-0.jpg".format(i * fr + 1))
		else :
			print("Turning ", file, " into ", "Temp{}".format(i))
			os.rename(file, dir + "/Temp{}.jpg".format(i))
			temp.append(dir + "/Temp{}.jpg".format(i))

	return temp
		
def my(text) :
	sub = text.split("Temp")
	if(len(sub) < 2) :
		sub = text.split("Style")
	rep = sub[1]
	rep = rep.replace(".jpg", "")
	rep = rep.replace("-", "")
	return int(rep)

def remove_files(remove, num_styles, style_frames) :
	for num in remove :
		if not num.isdigit() :
			print("NAN. Continuing...")
			continue
		else :
			name = "Style{}-0.jpg".format(int(num))
		if os.path.exists("temp/" + name):
			os.remove("temp/" + name)
			style_frames.remove(os.getcwd() + "/temp/" + name)
			num_styles -= 1
			print(name + " removed")
		else:
			print("The file does not exist")
	return num_styles, style_frames

def interpolate(input_files) :
	print("Interpolating... ")
	files = NST.run_interp(temp_folder, input_files)
	targets = get_interp_targets(input_files, files)
	files = Process.fade(targets, fr//num_styles, files)
	print("\n\n\n")
	print("Renaming files...")
	files = rename_files(temp_folder, fr, files, False)
	return files

def get_interp_targets(style_frames, files) :
	print("\nGetting style files")
	interp_targets = {}
	for i in range(len(style_frames)) :
		frame_number = files.index(style_frames[i])
		style_frame = style_frames[i]
		if(i == 0) : 
			previous_frame = files[(len(files) - 1)]
		else :
			previous_frame = files[frame_number - 1]
		interp_targets[style_frame] = previous_frame
	return interp_targets

def upscale_imgs(files) :
	print("Upscaling images... ")
	for i, file in enumerate(files) :
		out = results_folder + "/Frame{}.jpg".format(i)
		Process.process(file, out)
		print("Frame{}.jpg finished".format(i))

def generate_video():
	print("Generating video... ")
	loops = (duration * fr) // len(files)
	os.chdir(results_folder)

	for i in range(loops) :
		ffmpeg = "ffmpeg -framerate " + str(fr) + " -i Frame%d.jpg -c:v libx264 "+ out_name + "{}.mp4".format(i) 
		os.system(ffmpeg)


	os.system("(for %i in (*.mp4) do @echo file '%i') > mylist.txt")
	ffmpeg = "ffmpeg -f concat -safe 0 -i mylist.txt -c copy "+ out_name + ".mp4"
	os.system(ffmpeg)


res_folder_name = input("Enter name of output folder - ")

cwd = os.getcwd()
t = cwd + "/temp"
if os.path.exists(t) :
	shutil.rmtree(t)
results_path = os.getcwd() + "/Results/" + res_folder_name
if not os.path.exists(results_path) :
	results_folder = NST.create_results_folder("Results/" + res_folder_name)
else :
	results_folder = results_path

temp_folder = NST.create_results_folder("temp")
style_path = "Style/" + input("Enter style folder - ") + "/"
style_files = input("Enter style files - ")
if style_files != "" :
	style_files = style_files.split(" ")
	for i, file in enumerate(style_files) :
		file_name = style_path + style_files[i] + ".jpg"
		style_files[i] = file_name
else :
	style_files = glob.glob(style_path + "*.jpg")

content_path = "Content/" + input("Enter content folder - ") + "/"
content_files = input("Enter style files - ")
if content_files != "" :
	content_files = content_files.split(" ")
	for i, file in enumerate(content_files) :
		file_name = content_path + content_files[i] + ".jpg"
		content_files[i] = file_name
else :
	content_files = glob.glob(content_path + "*.jpg")

num_styles = len(style_files)
num_content = len(content_files)
form = "jpg"
fr =  prefs.fr
out_name = input("Enter output file name - ")
duration =  prefs.duration
NST.imgWidth = prefs.width
print("Transferring styles... ")
files = NST.run_styles(temp_folder, style_files, content_files)
files = rename_files(temp_folder, fr, files, True)
upscale_imgs(files)