import shutil
import NST
import os


content_img = "Content/" + input("Enter name of content image - ") + ".jpg"
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

num_styles = len(style_files)
form = "jpg"
fr =  prefs.fr
out_name = input("Enter output file name - ")
duration =  prefs.duration