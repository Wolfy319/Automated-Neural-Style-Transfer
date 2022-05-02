###
# Copyright 2022 Wolfy Fiorini
# All rights reserved
###

import requests
from PIL import Image
from PIL import ImageFilter

def process(input, out) : 
    image = scale(input, out)
    image = crop(image)
    image = sharpen(image, 2)
    image.save(out)



def scale(input, out) :
    r = requests.post(
        "https://api.deepai.org/api/waifu2x",
        files={
            'image': open(input, 'rb'),
        },
        headers={'api-key': 'eec8e06f-7015-44ba-8c46-a4a6295ab263'}
    )
    url = r.json()["output_url"]
    r = requests.get(url, allow_redirects=True)

    open(out, 'wb').write(r.content)
    img = Image.open(out)
    return img

def crop(image) :
    im = image.copy()
    width, height = im.size
    left = 20
    top = 20
    right = width - 20
    bottom = height - 20
    im = im.crop((left, top, right, bottom))
    return im

def sharpen(image, passes) :
    # Open an already existing image
    im = image.copy()
    for i in range(passes) :
        sharpened1 = im.filter(ImageFilter.SHARPEN)
        im = sharpened1
    return im

def resize(image, height, width) :
    image = Image.open(input)
    image = image.resize((height, width))
    image.save(input)
    return

def fade(images, num_between, files) :
    new_files = []

    for file in files :
        if images.__contains__(file) :
            im1 = Image.open(file)
            im2 = Image.open(images[file])
            name = images[file]
            name = name.replace(".jpg", "")
            num = findnum(name)
            print(num)
            if num == "1" :
                name = name.replace("Temp1-0", "")
                num = findnum(file)
                print("\n\n 1 found\n\n")
                print(num)
            name = name.replace("Temp{}-0".format(num), "")
            name += "Temp" + num
            save_name = name
            alpha = 1
            print("blending " + file + " with " + images[file])
            for i in range(num_between) :
                name = name + "-{}.jpg".format(i + 1)
                alpha -= 1 / (num_between)
                print(alpha)
                blend = Image.blend(im1, im2, alpha)
                blend.save(name)
                new_files.append(name)
                name = save_name
            new_files.append(file)
        else :
            new_files.append(file)
    return new_files

def findnum(name) :
    split = name.split("Temp")
    nums = split[1]
    nums = nums.split("-")
    return nums[0]