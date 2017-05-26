from PIL import Image
import sys

folder = 'data/'
filename = sys.argv[1]
level = sys.argv[2]

img = Image.open(folder + filename)

out = open(folder+'compressed/'+filename, 'w')

img.save(out, format = 'jpeg', quality=int(level))
