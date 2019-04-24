import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from scipy.misc import imsave
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 
import sys, glob, os, csv

use_cuda=True
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# using pretrain proxy model, ex. VGG16, VGG19...
model = resnet50(pretrained=True)
# or load weights from .pt file
#model = torch.load_state_dict(...)
# use eval mode
model.eval()

# loss criterion
loss = nn.CrossEntropyLoss()

input_dir = sys.argv[1]
f_names = glob.glob(input_dir + '*.png')

imgs = []
for i in range(len(f_names)):  # f_names为所有图片地址，list
    img = Image.open(input_dir + str(i).zfill(3) + '.png')  # 读取图片
    #img.show()
    arr_img = np.array(img)  # 图片转换为数组
    arr_img = np.expand_dims(arr_img, axis=0)   # 增加第一个batch维度
    imgs.append(arr_img) # 把图片数组加到一个列表里面
    print("loading no.%s image."%i)
x = np.concatenate([i for i in imgs]) # 把所有图片数组concatenate在一起
print(x.shape)

labels = np.load('labels.npy')
"""target_labels = np.zeros((len(labels), 1000))
target_labels[np.arange(labels.shape[0]), labels] = 1
print(target_labels.shape)
"""
epsilon = 0.08
for i in range(x.shape[0]):
    image = x[i]
    """# you can do some transform to the image, ex. ToTensor()
    trans = transform.Compose([transform.ToTensor()])
    
    image = trans(image)
    image = image.type('torch.FloatTensor')
    image = image.unsqueeze(0)"""
    image = image / 255.0
    trans = transform.Compose([transform.ToTensor()])
    normalize = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = trans(image)
    image = image.type('torch.FloatTensor')
    image = normalize(image)
    image = image.unsqueeze(0)
    image.requires_grad = True
    
    # set gradients to zero
    zero_gradients(image)
    
    output = model(image)
    argmax = np.argmax(output.detach().numpy())
    g = loss(output, torch.from_numpy(np.array(labels[i])).unsqueeze(0))
    g.backward() 
    
    # add epsilon to image
    image = image + epsilon * image.grad.sign_()

    image = image.squeeze(0)
    image = transform.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])(image)
    image = image.transpose(0,1).transpose(1,2)
    image = torch.clamp(image, 0, 1)
    image = image.detach().numpy()
    image = image * 255.0
    image = image.astype(np.uint8)
    image = np.clip(image, 0, 255)

    print('saving...' + str(i).zfill(3))
    # do inverse_transform if you did some transformation
    # image = transform.ToPILImage()(image[0])
    if not os.path.exists(sys.argv[2]) :
        os.makedirs(sys.argv[2])
    out_path = sys.argv[2] + str(i).zfill(3) + '.png'
    imsave(out_path,image)
    # image.save(out_path)