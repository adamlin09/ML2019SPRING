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
    mdfk = np.copy(M)
    mdfk -= np.min(mdfk)
    mdfk /= np.max(mdfk)
    mdfk = (mdfk * 255).astype(np.uint8)
    return mdfk

imgs, img_shape = load_data(sys.argv[1])
mean = np.mean(imgs, axis=0)
imgs -= mean

# report (a)
average = process(mean)
io.imsave('report1_a.jpg', average.reshape(img_shape))
print('report1_a : OK!')

# report (b)
U, s, V = np.linalg.svd(imgs.T, full_matrices=False)
U_eigen = U[:, :5]
U_eigen = U_eigen.T
for i in range(len(U_eigen)):
    eigen_vector = process(U_eigen[i])
    io.imsave(str(i) + '_eigenface.jpg', eigen_vector.reshape(img_shape))
print('report1_b : OK!')

# report (c)
IMAGE_PATH = 'Aberdeen'
test_image = ['1.jpg','10.jpg','22.jpg','37.jpg','72.jpg'] 
for x in test_image: 
    # Load image & Normalize
    print(x)
    picked_img = io.imread(os.path.join(IMAGE_PATH,x))  
    X = picked_img.flatten().astype('float32') 
    print(X)
    X -= mean
    
    # Compression
    weight = np.array(X.dot(U[:, :5]))  

    # Reconstruction
    reconstruct = process(weight.dot(U[:, :5].T) + mean)
    print(reconstruct)
    io.imsave(x[:-4] + '_reconstruction.jpg', reconstruct.reshape(img_shape)) 
print('report1_c : OK!')

# report (d)
for i in range(5):
    number = s[i] * 100 / sum(s)
    print(number)

end = time.time()
elapsed = end - start
print("Time taken: ", elapsed, "seconds.")