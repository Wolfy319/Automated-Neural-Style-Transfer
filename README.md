# Automated Neural Style Transfer
*Author - Wolfy Fiorini*<br></br>
A Python program to automatically generate high-res art from photos and style references in batches. 

This project was made over the course of about 2 months. At the beginning, I had no knowledge of AI or even Python in general, but this project gave me a fantastic foundation!

## Examples:
Content image:<br></br><img src="https://user-images.githubusercontent.com/60371754/221306020-ad3219bd-ebad-4eff-b305-4c26df83d3d9.jpg" alt="alt text" width="222" height="312"><br></br>
Style image:<br></br><img src="https://user-images.githubusercontent.com/60371754/221306292-71ddbc00-6c78-4290-bff8-979a395ade20.jpg" alt="alt text" width="300" height="300"><br></br>
Result:<br></br>
<img src="https://user-images.githubusercontent.com/60371754/221306949-33b57ffa-58d6-4d59-86cc-7d6c6280f89e.jpg" alt="alt text" width="444" height="624">
<br></br>
<hr>
<br></br>
Content image:<br></br><img src="https://user-images.githubusercontent.com/60371754/221307461-fb23f621-9260-48d8-95ad-c3ce7eef2712.jpg" alt="alt text" width="350" height="200"><br></br>
Style image:<br></br><img src="https://user-images.githubusercontent.com/60371754/221308712-74c0f43d-f997-43f8-bc29-13be3537daed.jpg" alt="alt text" width="300" height="300"><br></br>
Result:<br></br>
<img src="https://user-images.githubusercontent.com/60371754/221309016-383c5c67-fb40-4487-b3f1-086f80e6e57e.jpg" alt="alt text" width="600" height="400">
<br></br>
<hr>
<br></br>

# How to run
To run this application, you'll need to have Python installed on your computer, and an IDE to run the code. In addition, you'll need to install PyTorch locally (instructions - https://pytorch.org/get-started/locally/#windows-installation). Once you have this, follow these steps:
* Clone the repo into your IDE
* Create a ***Content*** folder and a ***Style*** folder in the same directory as the ***NST.py*** file
* Put any photos you want to apply styles to in the ***Content*** folder, and any pieces of art whose style you want them to be in in the ***Style*** folder.
* Run the ***driver.py*** file
* If you created folders inside the Content/ or Style/ directory (for example, seperate folders for texture photos and paintings), enter the name of the folders you want to use. If you don't have any subfolders, just press enter when prompted.
* Enter the name of the folder you would like the results to be placed in. If the name entered doesn't match any folder, one will be created automatically
* Sit back and wait! You'll be able to see the progress of each image in the console as it runs. The low resolution images will be put in a temp directory, and then upscaled and placed in the Results directory. Enjoy!

# About
This AI model was built using PyTorch, and implements the "Neural Style Transfer" method. The reason this method was chosen was because it requires no training data at all to replicate art styles.

I started this project to collaborate with a photographer friend and make art. The concept was to build an AI that could take any of his photos and turn it into a painting in any style we chose. However, most AI models that can transfer artstyles (such as CycleGAN) require massive amounts of training images, and will only be able to turn a single subject into a single artstyle - for example CycleGAN can be trained to take photos of dogs and turn them into Van Gogh pieces. This wouldn't work for me, since it would mean I couldn't use my friends photos, and would have to settle on one art style. So, I researched Neural style transfer!
