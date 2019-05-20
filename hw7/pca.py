import numpy as np
import sys, csv, glob, time, os
from skimage import io, img_as_float64, img_as_float, img_as_int

start = time.time()

def load_data(path):
    f_names = sorted(glob.glob(path + '*.jpg'), key=lambda x:int(x[9:-4]))
    img_shape = io.imread(f_names[0]).shape
    imgs = []
    for i in range(len(f_names)):  # f_names为所有图片地址，list
        img = io.imread(f_names[i]).astype('float')
        imgs.append(img.flatten()) # 把图片数组加到一个列表里面
        print('\r> Loading \'{}\''.format(f_names[i]), end="", flush=True)
    print('')
    imgs = np.array(imgs)
    return imgs, img_shape

def process(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M


imgs, img_shape = load_data(sys.argv[1])
mean = np.mean(imgs, axis=0)
imgs -= mean

# report (a)
average = process(mean)
io.imsave('report1_a.jpg', average.reshape(img_shape))
print('report1_a : OK!')

# report (b)
U, s, V = np.linalg.svd(imgs.T, full_matrices=False)
U = U.T
for i in range(5):
    eigen_vector = process(U[i])
    io.imsave(str(i) + '_eigenface.jpg', eigen_vector.reshape(img_shape))
print('report1_b : OK!')

# report (c)
test = [1, 10, 22, 37, 72]
U_new = U[:5].T #1080000, 5
s_new = np.diag(s)[:5]   #5, 415
V_new = V[:, test]  #415, 5
reconstruction = np.dot(U_new, s_new)
reconstruction = np.dot(reconstruction, V_new)
reconstruction = reconstruction.T
for i in range(len(reconstruction)):
    x = process(reconstruction[i] + mean)
    io.imsave(str(test[i]) + '_reconstruction.jpg', x.reshape(img_shape))
print('report1_c : OK!')

# report (d)
for i in range(5):
    number = s[i] * 100 / sum(s)
    print(number)

end = time.time()
elapsed = end - start
print("Time taken: ", elapsed, "seconds.")