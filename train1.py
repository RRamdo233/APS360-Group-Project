import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset, DataLoader

class T1Dataset(Dataset):
    def __init__(self, data, transform=None):
        # list of tuples (3d image arrays, AD label)
        self.data = data
        # labels.csv
        #self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        scan = torch.from_numpy(self.data[index][0]).float()
        y = self.data[index][1]

def prepare_dataset():
    runs_list = next(os.walk('./Data/Images'))[1]

    path = "./Data/Labels.csv"
    header = ['Run','AD']
    Labelsdf = pd.read_csv(path, names=header, usecols=[1,2], skiprows=1, index_col=False)
    labels_dict = dict(zip(Labelsdf.Run, Labelsdf.AD))

    low_bound = 100
    high_bound = 200
    img_scale = 4
    depth_scale = 4
    #list of tuples (image data, AD)
    all_data = []
    for run in runs_list:
        run_dir = os.path.join(img_dir, run)
        run_imgs = []
        for filename in os.listdir(run_dir):
            img_num = int(filename[-7:-4])
            if (100 <= img_num < 200):
                img_slice = cv2.imread(os.path.join(run_dir, filename), cv2.IMREAD_GRAYSCALE)
                #print(img_slice)
                img_slice = cv2.resize(img_slice, (0,0), fx=1/img_scale, fy=1/img_scale, interpolation=cv2.INTER_AREA)
                run_imgs.append(img_slice)

        temp_arr = np.array(run_imgs)

        if temp_arr.size == 0:
            print(run)
            continue

        final_slices = []
        #print(temp_arr.shape[2])

        for y in range(temp_arr.shape[2]):
            xz_pane = temp_arr[:, :, y]
            scaled_xz = cv2.resize(xz_pane, (0, 0), fy=1/depth_scale, fx=1, interpolation=cv2.INTER_AREA)
            final_slices.append(scaled_xz)

        final_array = np.dstack(final_slices)
        #print(final_array.shape)
        run_tuple = (final_array, labels_dict[run])
        all_data.append(run_tuple)

    return scan_dataset = T1Dataset(all_data, None)

def train(model, dataset, num_epochs=10, batch_size=32, learning_rate=1e-4):
    criterion = nn.BCELoss()
    # use Adam for CNN
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs, train_losses, val_losses, train_acc, val_acc = [], [], [], [], []

    train_loader = DataLoader(dataset, batch_size, shuffle=True)

    loss = 0

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epochs.append(epoch)
        train_losses.append(loss)

        print(f"Epoch: {epoch} Training Loss: {train_losses[-1]}")

    plt.title("Training Curve")
    plt.plot(epochs, train_losses, label="Train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Loss: {}".format(train_losses[-1]))

model = torch.load('./PretrainedModels/resnext-101-64f-kinetics-hmdb51_split1.pth')
