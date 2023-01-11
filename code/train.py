import csv
import os
import random
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import vgg_face_dag, Classification, ContrastiveLoss
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import math
from torch import optim
import warnings
import torch.nn.functional as F
from model import Facenet
from dataloader import FacenetDataset, dataset_collate
from functools import partial
from utils import show_config

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    gauss = cv2.GaussianBlur(image, (7, 7), 0)
    # gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
    # blur = cv2.blur(gauss, (3, 3))
    # cv2.imshow('gray', gray)
    # cv2.waitKey(10000)
    return gauss


def add_peppersalt_noise(image, n=1000):  # 生成椒盐噪声
    result = image.copy()
    w, h = image.shape[:2]
    # 生成n个椒盐噪声
    for i in range(n):
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        if np.random.randint(0, 2) == 0:
            result[x, y] = 0
        else:
            result[x, y] = 255
    return result


def triplet_loss(alpha=0.2):
    def _triplet_loss(y_pred, Batch_size):
        anchor, positive, negative = y_pred[:int(Batch_size)], y_pred[int(Batch_size):int(2 * Batch_size)], y_pred[
                                                                                                            int(2 * Batch_size):]

        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

        keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()
        hard_triplets = np.where(keep_all == 1)

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
        return loss

    return _triplet_loss


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
                                              ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, Batch_size, fp16, scaler, save_period, local_rank):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0

    if local_rank == 0:
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs1, outputs2 = model_train(images, "train")

            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs1, outputs2 = model_train(images, "train")

                _triplet_loss = loss(outputs1, Batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                _loss = _triplet_loss + _CE_loss
            scaler.scale(_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

        total_triple_loss += _triplet_loss.item()
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                'total_CE_loss': total_CE_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

            optimizer.zero_grad()
            outputs1, outputs2 = model_train(images, "train")

            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss

            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            val_total_triple_loss += _triplet_loss.item()
            val_total_CE_loss += _CE_loss.item()
            val_total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_total_triple_loss': val_total_triple_loss / (iteration + 1),
                                'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                                'val_accuracy': val_total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))


def train_dataset():
    fp16, Cuda = False, True
    print('Cuda:', Cuda)
    annotation_path = "../init_data/train/cls_train.txt"
    input_shape = [160, 160, 3]
    model_path = './model/facenet_mobilenet.pth'
    pretrained = False
    batch_size, Init_Epoch, Epoch, Init_lr = 18, 0, 15, 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum, weight_decay, save_period, local_rank = 0.9, 0, 1, 0
    lr_decay_type = "cos"
    with open(annotation_path) as f:
        dataset_path = f.readlines()
    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    model = Facenet(num_classes=num_classes, pretrained=pretrained)

    if local_rank == 0:
        print('Load weights {}.'.format(model_path))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    loss = triplet_loss()
    scaler = None

    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    val_split = 0.01
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    show_config(
        num_classes=num_classes, backbone="mobilenet", model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
        save_period=save_period, num_train=num_train, num_val=num_val
    )

    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_classes, random=True)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], num_classes, random=False)

        train_sampler = None
        val_sampler = None
        shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size // 3,
                         pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size // 3,
                             pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        for epoch in range(Init_Epoch, Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen,
                          gen_val, Epoch, Cuda, batch_size // 3, fp16, scaler, save_period
                          , local_rank)
            torch.save(model.state_dict(), os.path.join('model/facenet_mobilenet_test.pth'))


def train():
    print(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    train_path = '../init_data/train/'
    files_list = os.listdir(train_path + 'data')
    # print(files_list)
    len_files = len(files_list)
    len_mid = int(len_files * 0.8)
    random.shuffle(files_list)
    train_file = files_list[:len_mid]
    val_file = files_list[len_mid:]
    # print('train:', len(train_file), 'val:', len(val_file))
    data_csv = open(train_path + 'annos.csv', "r")
    reader = csv.reader(data_csv)
    labels = {}
    for item in reader:
        if reader.line_num == 1:
            continue
        labels[item[0]] = item[1]
    # print(labels)
    train_file += val_file
    # print(len(train_file))
    train_data, val_data = [], []
    train_file_bar = tqdm(train_file, file=sys.stdout)
    for file in train_file_bar:
        img1 = Image.open(train_path + 'data/' + file + '/a.jpg')
        img2 = Image.open(train_path + 'data/' + file + '/b.jpg')
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        # faces = face_detector.detectMultiScale(img_gray)
        # img1_clip = border_clipping(img1_array)
        # img2_clip = border_clipping(img2_array)
        # gauss = cv2.GaussianBlur(img1_array, (5, 5), 1, 2)
        # median = cv2.medianBlur(img1_array, 5)
        # blur = cv2.blur(img1_clip, (5, 5))
        # grass_gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
        # median_gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        # blur_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", cv2.cvtColor(img1_array, cv2.COLOR_BGR2GRAY))
        # cv2.waitKey(10000)
        # cv2.imshow("grass_gray", grass_gray)
        # cv2.waitKey(10000)
        # cv2.imshow("median_gray", median_gray)
        # cv2.waitKey(10000)
        # cv2.imshow("blur_gray", blur_gray)
        # cv2.waitKey(10000)
        # gray_img1 = cv2.cvtColor(img1_clip, cv2.COLOR_BGR2GRAY)
        # img1_blur = cv2.blur(gray_img1, (5, 5))
        # gray_img2 = cv2.cvtColor(img2_clip, cv2.COLOR_BGR2GRAY)
        # img3_blur = cv2.flip(img1_blur, 1)
        # img4_clip = cv2.flip(gray_img2, 1)
        # cv2.imshow("gray_img1", blur)
        # cv2.waitKey(10000)
        # cv2.imshow("gray_img2", gray_img2)
        # cv2.waitKey(10000)
        # img1 = Image.fromarray(np.uint8(img1_blur))
        # img3 = Image.fromarray(np.uint8(img3_blur))
        # img2 = Image.fromarray(np.uint8(gray_img2))
        # img4 = Image.fromarray(np.uint8(img4_clip))
        img1_crop, img2_crop = Cropping(img1_array), Cropping(img2_array)
        # cv2.imshow('img2', img1_crop)
        # cv2.waitKey(10000)
        # pepper_img2 = add_peppersalt_noise(img2_crop)
        # cv2.imshow('gauss', pepper_img2)
        # cv2.waitKey(10000)
        salt_img2 = add_peppersalt_noise(img2_crop)
        # cv2.imshow('salt_img2', salt_img2)
        # cv2.waitKey(10000)
        pro_img1 = Preprocessing(img1_crop)
        # cv2.imshow('1', pro_img1)
        # cv2.waitKey(10000)
        pro_img2 = Preprocessing(salt_img2)
        # cv2.imshow('pro_img2', pro_img2)
        # cv2.waitKey(10000)
        # cv2.waitKey(10000)
        img3 = cv2.flip(img1_crop, 1)
        img4 = cv2.flip(img2_crop, 1)
        img1 = Image.fromarray(np.uint8(pro_img1))
        img3 = Image.fromarray(np.uint8(img3))
        img2 = Image.fromarray(np.uint8(pro_img2))
        img4 = Image.fromarray(np.uint8(img4))
        img1, img2, img3, img4 = transform(img1), transform(img2), transform(img3), transform(img4)
        train_data.append(tuple([img2, img1, torch.tensor([int(labels[file])], dtype=torch.long)]))
        train_data.append(tuple([img4, img1, torch.tensor([int(labels[file])], dtype=torch.long)]))
        train_data.append(tuple([img4, img3, torch.tensor([int(labels[file])], dtype=torch.long)]))
        train_data.append(tuple([img2, img3, torch.tensor([int(labels[file])], dtype=torch.long)]))
        train_file_bar.desc = "Process Training Files"
    # val_file_bar = tqdm(val_file, file=sys.stdout)
    # for file in val_file_bar:

    #     img1 = Image.open(train_path + 'data/' + file + '/a.jpg')
    #     img2 = Image.open(train_path + 'data/' + file + '/b.jpg')
    #     img1_array = np.array(img1)
    #     img2_array = np.array(img2)
    #     img1_crop, img2_crop = Cropping(img1_array), Cropping(img2_array)
    #     pepper_img2 = add_peppersalt_noise(img2_crop)
    #     pro_img2 = Preprocessing(pepper_img2)
    #     pro_img1 = Preprocessing(img1_crop)
    #     # img1_clip = border_clipping(img1_array)
    #     # img2_clip = border_clipping(img2_array)
    #     # gray_img1 = cv2.cvtColor(img1_clip, cv2.COLOR_BGR2GRAY)
    #     # blur = cv2.blur(gray_img1, (5, 5))
    #     # gray_img2 = cv2.cvtColor(img2_clip, cv2.COLOR_BGR2GRAY)
    #     img1 = Image.fromarray(np.uint8(pro_img1))
    #     img2 = Image.fromarray(np.uint8(pro_img2))
    #     img1, img2 = transform(img1), transform(img2)
    #     val_data.append(tuple([img1, img2, torch.tensor([float(labels[file])])]))
    #     val_file_bar.desc = "Process Validation Files"
    # print(len(train_data), len(val_data))
    # print(train_data[0])
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=16)
    # val_loader = DataLoader(dataset=val_data, shuffle=False, batch_size=16)
    vgg_net = vgg_face_dag(weights_path="./model/VGG Face")
    vgg_net.to(device)
    for param in vgg_net.parameters():
        param.requires_grad = False
    net = Classification().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(net, lr=0.001, betas=(0.9, 0.999))

    epochs = 50
    vgg_save_path = './model/train_net.pth'
    save_path = './model/train_net.pth'
    train_steps = len(train_loader)
    # best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        true_sum = 0
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            img1, img2, label = data
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            vgg_output, output = vgg_net(img1, img2)
            # print(vgg_output)
            output = net(output)
            # output1, output2 = [], []
            # for i in output.detach():
            #     output1.append(i[0].item())
            #     output2.append(i[1].item())
            # output1, output2 = torch.tensor(output1).unsqueeze(-1), torch.tensor(output2).unsqueeze(-1)
            # print(output)
            # euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            # print(euclidean_distance)
            # print(output)
            # print(out, label.unsqueeze(1).to(device))
            # print(label.squeeze(-1))
            label = label.squeeze(-1)
            # print(output)
            loss = criterion(output, label)
            # print(torch.max(output, 1))
            _, pred = torch.max(output, 1)
            # print(pred)
            # print(label)
            # loss = criterion(output1, output2, label)
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct = pred.eq(label.view_as(pred))
            # print(correct)
            for i in correct:
                if i.item() == True:
                    true_sum += 1
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        print('[epoch %d] train_loss: %.3f acc：%.3f' % (
            epoch + 1, running_loss / train_steps, true_sum / len(train_data)))
        # net.eval()
        # acc = 0.0
        # with torch.no_grad():
        #     val_bar = tqdm(val_loader, file=sys.stdout)
        #     for data in val_bar:
        #         img1, img2, label = data
        #         output1, output2 = net(img1.to(device), img2.to(device))
        #         # print(output1, output2)
        #         # print(label)
        #         outs = F.pairwise_distance(output1, output2)
        #         # outs = F.cross_entropy(output1, output2)
        #         # print(outs)
        #         # print(res)
        #         # print(label)x
        #         for i, out in enumerate(outs):
        #             similar = 1 / (1 + out.item())
        #             # similar = (1 + out) / 2
        #             similar = round(similar, 1)
        #             print(similar, label[i][0])
        #             if similar > 0.5:
        #                 res = 1.0
        #             else:
        #                 res = 0.0
        #             if res == label[i][0]:
        #                 acc += 1
        #         val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
        #                                                    epochs)
        # val_accurate = acc / len(val_data)
        # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #       (epoch + 1, running_loss / train_steps, val_accurate))
        #
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
    torch.save(net.state_dict(), save_path)
    torch.save(vgg_net.state_dict(), vgg_save_path)
    # print("beat acc:", best_acc)


if __name__ == '__main__':
    train_dataset()
    print("train dataset finish!")
    train()
    print("train data finish!")
