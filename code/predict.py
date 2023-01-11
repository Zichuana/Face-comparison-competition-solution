import os, sys
import cv2
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import torch
from utils import detect_image
from torch.utils.data import DataLoader
from model import vgg_face_dag


def Cropping(image):
    row = image.shape[0]
    col = image.shape[1]
    # print("x=", row, " y=", col)
    left = row
    top = col
    right = 0
    bottom = 0
    for r in range(row):
        for c in range(col):
            if image[r][c][0] < 255 and image[r][c][0] != 0:
                if top > r:
                    top = r
                if bottom < r:
                    bottom = r
                if left > c:
                    left = c
                if right < c:
                    right = c
    # print(left, top, right, bottom)
    image = image[top:bottom, left:right]
    return image


def Preprocessing(image):
    blur = cv2.blur(image, (5, 5))
    return blur


def predict(file1, file2):
    with torch.no_grad():
        test_data = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        img1 = Image.open(file1)
        img2 = Image.open(file2)
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        img1_clip, img2_clip = Cropping(img1_array), Cropping(img2_array)
        gray1 = cv2.cvtColor(img1_clip, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_clip, cv2.COLOR_BGR2GRAY)
        pro_img1 = Preprocessing(gray1)
        pro_img2 = Preprocessing(gray2)
        img1 = Image.fromarray(np.uint8(pro_img1))
        img2 = Image.fromarray(np.uint8(pro_img2))
        facenet_distance = detect_image(img1, img2)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './model/train_net.pth').replace('\\', '/')
        vgg_net = vgg_face_dag()
        vgg_net.load_state_dict(torch.load(model_path, map_location=device))
        vgg_net.to(device)
        pro_img1 = Preprocessing(img1_clip)
        pro_img2 = Preprocessing(img2_clip)
        img1 = Image.fromarray(np.uint8(pro_img1))
        img2 = Image.fromarray(np.uint8(pro_img2))
        img1, img2 = transform(img1), transform(img2)
        test_data.append(tuple([img1, img2]))
        # print(test_data)
        test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=1)
        for test in test_loader:
            # print(test)
            input1, input2 = test
            input1, input2 = input1.to(device), input2.to(device)
        train_distance, _ = vgg_net(input1, input2)
    return train_distance.item(), facenet_distance[0]


if __name__ == "__main__":
    x1, x2 = predict('../init_data/train/data/2/a.jpg', '../init_data/train/data/2/b.jpg')
    print(x1, x2)
    print(1/(1+x1), 1/(1+x2))
    res = 0.01/(1+x1)+0.99/(1+x2)
    print(res)
