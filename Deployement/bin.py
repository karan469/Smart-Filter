import numpy as np

im = (np.random.rand(10,10)*255).astype('int32')
bg = (np.random.rand(10,10)*255).astype('int32')
mask = (np.random.rand(10,10)*2).astype('int32')

print(im)
print(bg)
print(mask)
pre_im = (np.where(mask==1, im, bg))
