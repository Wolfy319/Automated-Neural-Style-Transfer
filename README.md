## Automated Neural Style Transfer
Python program to automatically generate high-res art from photos and style references. This AI model was built using PyTorch, and implements the "Neural Style Transfer" method. The reason this method was chosen was because it requires no training data at all to replicate art styles.

This project was made over the course of about 2 months. At the beginning, I had no knowledge of AI or even Python in general, but this project gave me a fantastic foundation!

# Examples:
Content image:
<img src="https://user-images.githubusercontent.com/60371754/221306020-ad3219bd-ebad-4eff-b305-4c26df83d3d9.jpg" alt="alt text" width="111" height="156">
Style image:
<img src="https://user-images.githubusercontent.com/60371754/221306292-71ddbc00-6c78-4290-bff8-979a395ade20.jpg" alt="alt text" width="100" height="100">
Result:
<img src="https://user-images.githubusercontent.com/60371754/221306949-33b57ffa-58d6-4d59-86cc-7d6c6280f89e.jpg" alt="alt text" width="111" height="156">
<br></br>

Content image:
![wheat](https://user-images.githubusercontent.com/60371754/221307461-fb23f621-9260-48d8-95ad-c3ce7eef2712.jpg)
Style image:
![water-abstract-texture-number-pattern-reflection-1042640-pxhere com](https://user-images.githubusercontent.com/60371754/221308712-74c0f43d-f997-43f8-bc29-13be3537daed.jpg)
Result:
![Frame67](https://user-images.githubusercontent.com/60371754/221309016-383c5c67-fb40-4487-b3f1-086f80e6e57e.jpg)

# How to run
To run this application, you'll need to have Python installed on your computer, and an IDE to run the code. In addition, you'll need to install PyTorch locally (instructions - https://pytorch.org/get-started/locally/#windows-installation). Once you have this, follow these steps:
1. Clone the repo into your IDE
2. Create a Content folder and a Style folder in the same directory as the NST.py file
3. Put any photos you want to apply styles to in the Content folder, and any pieces of art whose style you want them to be in in the Style folder.
  - All styles will be applied to ALL photos. So, if you put 4 content photos in and 4 style references, you'll get 4x4 = 16 image results
  - Currently, the program will only run using .jpg files 
  - After experimenting, I've found that the best content images tend to be well lit and in a high resolution. The best style references tend to be ones that have lots of texture and color. 
  - Feel free to experiment! This program is pretty great at picking up on textures and patterns, so don't hesitate to use photos as a style reference. I've gotten great results from pictures of bubbles, photos of fur, foliage, brick walls, etc.
4. Look at the myconfig.py file and change the image width to fit your needs
  - The result photos will be 4x the size of what you put here (ex: with width 1000, the result will be 4000px wide)
  - The bigger the number, the longer the images will take to generate. If you have a GPU or a fast PC, 500-1000px wide should work. If you are running the program on a laptop or weaker PC, 300-500px will run much faster.
5. If you want the program to generate a short video from the results, uncomment all of the code at the bottom of driver.py
6. Run the driver.py file
7. If you created folders inside the Content/ or Style/ directory (for example, seperate folders for texture photos and paintings), enter the name of the folders you want to use. If you don't have any subfolders, just press enter when prompted.
8. Enter the name of the folder you would like the results to be placed in. If the name entered doesn't match any folder, one will be created automatically
9. Sit back and wait! You'll be able to see the progress of each image in the console as it runs. The low resolution images will be put in a temp directory, and then upscaled and placed in the Results directory. Enjoy!

I started this project to collaborate with a photographer friend and make art. The concept was to build an AI that could take any of his photos and turn it into a painting in any style we chose. However, most AI models that can transfer artstyles (such as CycleGAN) require massive amounts of training images, and will only be able to turn a single subject into a single artstyle - for example CycleGAN can be trained to take photos of dogs and turn them into Van Gogh pieces. This wouldn't work for me, since it would mean I couldn't use my friends photos, and would have to settle on one art style. So, I researched Neural style transfer!
