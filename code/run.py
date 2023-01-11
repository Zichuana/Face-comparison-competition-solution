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
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        # pro_img2 = Preprocessing(gray2)
        img1 = Image.fromarray(np.uint8(pro_img1))
        img2 = Image.fromarray(np.uint8(gray2))
        facenet_distance = detect_image(img1, img2)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/train_net.pth').replace('\\', '/')
        pro_img1 = Preprocessing(img1_clip)
        pro_img2 = Preprocessing(img2_clip)
        img1 = Image.fromarray(np.uint8(pro_img1))
        img2 = Image.fromarray(np.uint8(pro_img2))
        vgg_net = vgg_face_dag()
        vgg_net.load_state_dict(torch.load(model_path, map_location=device))
        vgg_net.to(device)
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


def main(to_pred_dir, result_save_path):
    subdirs = os.listdir(to_pred_dir)  # name
    train_distances, facenet_distances, labels = [], [], []
    for subdir in subdirs:
        x1, x2 = predict(os.path.join(to_pred_dir, subdir, "a.jpg"), os.path.join(to_pred_dir, subdir, "b.jpg"))
        # distances.append(result)
        train_distances.append(x1)
        facenet_distances.append(x2)
    min1 = min(train_distances)
    min2 = min(facenet_distances)
    max1 = max(train_distances)
    max2 = max(facenet_distances)
    for i, j in zip(train_distances, facenet_distances):
        x1 = (max1 - min1) / (max1 - min1 * 2 + i)
        x2 = (max2 - min2) / (max2 - min2 * 2 + j)
        res = x2 * 0.99 + x1 * (1 - 0.99)
        labels.append(round(res, 3))
        # res = 0.3*1/(1+x1)+0.7*(1+x2)
        # labels.append(round(res, 3))
    # dis_min = min(distances)
    # dis_max = max(distances)
    # for i in distances:
    #     res = 1 / (i + 1)
    #     labels.append(round(res, 3))
    fw = open(result_save_path, "w")
    fw.write("id,label\n")
    for subdir, label in zip(subdirs, labels):
        fw.write("{},{}\n".format(subdir, label))
    fw.close()


if __name__ == "__main__":
    # for i in range(245, 251):
    #     x1, x2 = predict('C:\\Users\\Zichuana\\Desktop\\计挑赛\\Model_Fusion_Project\\init_data\\train\data\\{}\\a.jpg'.format(i),
    #                  'C:\\Users\\Zichuana\\Desktop\\计挑赛\\Model_Fusion_Project\\init_data\\train\\data\\{}\\b.jpg'.format(i))
    #     print('{}'.format(i), x2)
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)
