import requests
from PIL import Image
from PIL import ImageFilter

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