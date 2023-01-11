import numpy as np
import torch
from PIL import Image
import os
from model import Facenet
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def detect_image(image_1, image_2):
    with torch.no_grad():
        print('device', device)
        net = Facenet(mode="predict").eval()
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/facenet_mobilenet_test.pth').replace(
            '\\', '/')
        net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        net.to(device)
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net.to(device)
        image_1 = resize_image(image_1, [160, 160],
                               letterbox_image=True)
        image_2 = resize_image(image_2, [160, 160],
                               letterbox_image=True)

        photo_1 = torch.from_numpy(
            np.expand_dims(np.transpose((np.array(image_1, np.float32)) / 255.0, (2, 0, 1)), 0))
        photo_2 = torch.from_numpy(
            np.expand_dims(np.transpose((np.array(image_2, np.float32)) / 255.0, (2, 0, 1)), 0))

        photo_1 = photo_1.to(device)
        photo_2 = photo_2.to(device)

    output1 = net(photo_1).cpu().numpy()
    output2 = net(photo_2).cpu().numpy()
    l1 = np.linalg.norm(output1 - output2, axis=1)

    return l1


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
