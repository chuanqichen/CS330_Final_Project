import argparse

import os
import random
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from load_csv import DataGenerator
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter
import torchvision
import unittest

def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer0 = torch.nn.LSTM(39, 1, batch_first=True)
        self.layer1 = torch.nn.LSTM(num_classes + 75 #2925/39 
                                    ,hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer0)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 2925] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####        
        B, K, N, I = input_images.shape[0], input_images.shape[1]-1,\
                     input_images.shape[2], input_images.shape[3]
        labels = torch.clone(input_labels)
        labels[:, K, :] = 0  # set the query set label to be 0 
        input_images = input_images.reshape(B, (K+1)*N*75, 39)
        input_images, _ = self.layer0(input_images)
        input_images = input_images.reshape(B, (K+1), N, 75)        
        x = torch.cat((input_images, labels), axis=3) 
        x = x.reshape(B, (K+1)*N, N+75) 
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)        
        x = x.reshape(B, K+1, N, N)
        return x
        #############################

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################
        #### YOUR CODE GOES HERE ####       
        labels_ = torch.clone(labels)
        preds = preds[:, -1:].squeeze(1).reshape(-1, self.num_classes)
        labels_ = labels_[:, -1:].squeeze(1).reshape(-1, self.num_classes).argmax(axis=1)
        return nn.CrossEntropyLoss()(preds, labels_)
        #############################


def train_step(images, labels, model, optim, eval=False):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()

def compute_acc(prediction, l, config):
    prediction = torch.reshape(
        prediction, [-1, config.num_shot + 1, config.num_classes, config.num_classes]
    )
    prediction = torch.argmax(prediction[:, -1, :, :], axis=2)
    l = torch.argmax(l[:, -1, :, :], axis=2)
    acc = prediction.eq(l).sum().item() / (config.meta_batch_size * config.num_classes)
    return acc

def main(config):
    print(config)
    log_dir = config.log_dir    
    os.makedirs(log_dir, exist_ok=True)        
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(log_dir=log_dir+ "/mann/"
        "classes"+str(config.num_classes)+ "_shots" +str(config.num_shot)+
        "_hdim" + str(config.hidden_dim) + "_learning_rate_" +str(config.learning_rate) 
        + "_run" + str(config.random_seed)
    )

    # Create Data Generator
    train_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="train",
        config=config,
        device=device,
        cache=config.image_caching,
    )
    train_loader = iter(
        torch.utils.data.DataLoader(
            train_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    test_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="test",
        config=config,        
        device=device,
        cache=config.image_caching,
    )
    test_loader = iter(
        torch.utils.data.DataLoader(
            test_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    # Create model
    model = MANN(config.num_classes, config.num_shot + 1, config.hidden_dim)
    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    import time

    times = []
    for step in range(config.train_steps):
        ## Sample Batch
        t0 = time.time()
        i, l = next(train_loader)
        i, l = i.to(device), l.to(device)
        t1 = time.time()

        ## Train
        pred, ls = train_step(i, l, model, optim)
        t2 = time.time()
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])

        ## Evaluate
        if (step + 1) % config.eval_freq == 0:
            train_acc = compute_acc(pred, l, config)
            print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, tl = next(test_loader)
            i, tl = i.to(device), tl.to(device)
            tpred, tls = train_step(i, tl, model, optim, eval=True)
            print("Train Loss:", ls.cpu().numpy(), "Test Loss:", tls.cpu().numpy())
            writer.add_scalar("Loss/test", tls, step)
            test_acc = compute_acc(tpred, tl, config)
            print("Train Accuracy:", train_acc, "\tTest Accuracy", test_acc)
            writer.add_scalar("Accuracy/train", train_acc, step)
            writer.add_scalar("Accuracy/test", test_acc, step)

            times = np.array(times)
            print(f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}")
            times = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="runs/",
                        help='directory to save to or load from')        
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_shot", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_steps", type=int, default=25000)
    parser.add_argument("--image_caching", type=bool, default=True)
    parser.add_argument("--data_folder", type=str, help="single csv file name or data folder", default="./data/")
    main(parser.parse_args())
