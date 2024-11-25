import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, sample, choice, random
import os
import pandas as pd
import seaborn as sn
import time

latin_names = [
    "Dascyllus reticulatus",
    "Plectroglyphidodon dickii",
    "Chromis chrysura",
    "Amphiprion clarkii",
    "Chaetodon lunulatus",
    "Chaetodon trifascialis",
    "Myripristis kuntee",
    "Acanthurus nigrofuscus",
    "Hemigymnus fasciatus",
    "Neoniphon sammara"
]

fish_class_to_latin_name = {
    f"fish_{(i+1):02}" : latin for i,latin in enumerate(latin_names)
}

TS = time.strftime("%Y%m%d_%H%M%S")

config = {
    "device": "cuda",
    "out_dir" : Path(__file__).parent / "out" / f"train_{TS}",
    "dataset_dir": Path(__file__).parent / "dataset" / "fish_new_2",
    "testing_dir": Path(__file__).parent / "out" / f"test_{TS}",
    "learning_rate": 0.0001,
    "batch_size": 64,
    "weight_decay": 0.0005,
    "CL_alpha": 0.5,
    "CL_beta": 0.5,
    "CL_margin": 1,
    "lrs_step_size": 20,
    "lrs_gamma": 0.5,
    "stop_lr": 0.000001
}

def create_target_to_indices(dataset: ImageFolder):
    """
    return a dictionary that maps from target/class to idx
    """
    target_to_indices = {}
    for idx in range(len(dataset.targets)):
        target = dataset.targets[idx]
        if target in target_to_indices:
            target_to_indices[target].append(idx)
        else:
            target_to_indices[target] = [idx]
    return target_to_indices
    
class FishPairsDataset(Dataset):
    
    def __init__(self, fish_dataset, transform, pairs):
        self.fish_dataset = fish_dataset
        self.transform = transform
        self.pairs = pairs  
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        idx1, idx2 = self.pairs[index]
        
        img1,l1 = self.fish_dataset.__getitem__(idx1)
        img2,l2 = self.fish_dataset.__getitem__(idx2)
        
        label = 1 if l1 == l2 else 0
        
        return self.transform(img1), self.transform(img2), label
    
class TwoRandomFishesDataset(Dataset):
    
    def __init__(self, fish_dataset: ImageFolder, transform, size):
        self.fish_dataset = fish_dataset
        self.transform = transform
        self.size = size
        self.target_to_indices = create_target_to_indices(fish_dataset)
        self.targets = [i for i in range(len(fish_dataset.classes))]
        
    def __len__(self):
        return self.size
    
    def _get_two_random_idx(self):   
        # Pick two fish types t1, t2
        # 50% of the time it picks the same type
        if random() > 0.5:
            t1, t2 = np.random.choice(self.targets, 2, replace = False)
        else:
            t1 = choice(self.targets)
            t2 = t1
            
        idx1 = choice(self.target_to_indices[t1])
        idx2 = choice(self.target_to_indices[t2])
        
        while idx1 == idx2: # Make sure picked indices are not equal
            idx2 = choice(self.target_to_indices[t2])
            
        return idx1, idx2
        
    def __getitem__(self, _):
        idx1, idx2 = self._get_two_random_idx()
        
        img1,l1 = self.fish_dataset.__getitem__(idx1)
        img2,l2 = self.fish_dataset.__getitem__(idx2)
        
        label = 1 if l1 == l2 else 0
        
        return self.transform(img1), self.transform(img2), label

def create_all_possible_pairs(num_indices):
    """
    Return list of all possible pairs of indices (i, j) where 
    - i, j < num_indices
    - i < j
    """
    pairs = []
    for i in range(num_indices):
        for j in range(i+1, num_indices):
            pairs.append((i, j))

    return pairs
        
class VGG11Siamese(nn.Module):
    
    def __init__(self, feature_vector_size) -> None:
        super().__init__()
        self.net = vgg.vgg11(weights = vgg.VGG11_Weights.IMAGENET1K_V1)
    
        # Replace classifier with custom so that it outputs the feature vectors
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_vector_size),
        )
        
    def forward(self, img1, img2):
        x1: torch.Tensor = self.net(img1)
        x2: torch.Tensor = self.net(img2)
        
        distance = (x2 - x1).pow(2).sum(1).sqrt() # Euclidean distance between feature vectors
        return distance
    

class ShapePrinter(nn.Module):

    def __init__(self, text = "") -> None:
        super().__init__()
        self.text = text

    def forward(self, x):
        print(self.text, x.size())
        return x

class CustomSiameseNetwork(nn.Module):
    
    def __init__(self, feature_vector_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(3),
            #ShapePrinter("Shape of input:"),
            nn.Conv2d(3, 42, 7),
            nn.ReLU(inplace = True),
            #ShapePrinter("Shape after first convolution layer:"),

            nn.MaxPool2d(6, 4),
            #ShapePrinter("Shape after first max pool:"),
            
            nn.Conv2d(42, 120, 4),
            nn.ReLU(inplace = True),
            #ShapePrinter("Shape after second convolution layer:"),

            nn.MaxPool2d(4, 4),
            #ShapePrinter("Shape after second max pool:"),

            nn.Conv2d(120, 80, 4),
            nn.ReLU(inplace = True),
            #ShapePrinter("Shape after third convolution layer:"),

            nn.MaxPool2d(4, 4),
            #ShapePrinter("Shape after third max pool:"),
        )
        
        self.feature_vector_extractor = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, feature_vector_size),
        )
    
    def forward(self, img1, img2):
        x1 = torch.flatten(self.conv(img1), start_dim = 1)
        x2 = torch.flatten(self.conv(img2), start_dim = 1)
        
        x1 = self.feature_vector_extractor(x1)
        x2 = self.feature_vector_extractor(x2)
        
        distance = (x2 - x1).pow(2).sum(1).sqrt() # Euclidean distance between feature vectors
        return distance
        

class ContrastiveLoss(nn.Module):
    
    def __init__(self, alpha, beta, margin):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, y, dw):
        return self.alpha * y * dw.pow(2) + (1 - y) * self.beta * torch.clamp(self.margin - dw, min = 0.0).pow(2)

def get_confusion_matrix_stats(confusion_matrix: np.ndarray):
    tp = confusion_matrix[0,0]
    fn = confusion_matrix[0,1]
    fp = confusion_matrix[1,0]
    tn = confusion_matrix[1,1]

    recall = tp / (tp + fn)
    precission = tp / (tp + fp)
    specificity = tn / (tn + fp)

    accuracy = confusion_matrix.trace() / confusion_matrix.sum()
    f1 = 2 * (precission * recall) / (precission + recall)

    return accuracy, precission, recall, specificity, f1, tp, fn, fp, tn

def create_confusion_matrix(true_labels, predicted_labels):
    confusion_matrix = np.zeros((2, 2))
    for i in range(true_labels.shape[0]):
        if true_labels[i] == 1:
            if predicted_labels[i] == 1:
                confusion_matrix[0, 0] += 1 # True positive
            else:
                confusion_matrix[0, 1] += 1 # False negative
        else: 
            if predicted_labels[i] == 1:
                confusion_matrix[1, 0] += 1 # False positive
            else:
                confusion_matrix[1, 1] += 1 # True negative
    return confusion_matrix

def train(model, data_loader, optimizer, loss_fn):
    loss_sum = 0
    device = config["device"]
    model.train(True)
    confusion_matrix = np.zeros((2, 2))
    num_rows = 0
    for batch in tqdm(data_loader, desc = "Training", leave = False):
        img1s, img2s, true_labels = batch
        
        img1s = img1s.to(device)
        img2s = img2s.to(device)
        true_labels = true_labels.to(device)
        
        optimizer.zero_grad()
        
        distance = model(img1s, img2s)
        predicted_labels = torch.where(
            distance < 0.5, 
            torch.ones_like(distance), 
            torch.zeros_like(distance)).type(torch.int8)
        
        confusion_matrix += create_confusion_matrix(true_labels, predicted_labels)
        
        # loss
        loss = loss_fn(true_labels, distance)
        
        # backward 
        loss.mean().backward()
        
        optimizer.step()
    
        loss_sum += loss.sum().item()
        num_rows += true_labels.size(0)
    
    accuracy, precission, recall, specificity, f1, tp, fn, fp, tn = get_confusion_matrix_stats(confusion_matrix)

    return {
        "loss": loss_sum,
        "avg_loss": loss_sum / num_rows,
        "accuracy": accuracy,
        "precission": precission,
        "recall": recall,
        "specificity": specificity,
        "F1": f1,
        "tp": tp,
        "fn": fn,
        "fp":fp,
        "tn": tn,
    }

    
def validate(model, data_loader, loss_fn):
    device = config["device"]
    loss_sum = 0
    confusion_matrix = np.zeros((2, 2))
    num_rows = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc = "Validation", leave = False):
            img1s, img2s, true_labels = batch
        
            # Move to gpu
            img1s = img1s.to(device)
            img2s = img2s.to(device)
            true_labels = true_labels.to(device)

            # Pass through model
            distance = model(img1s, img2s)

            predicted_labels = torch.where(
                distance < 0.5, 
                torch.ones_like(distance), 
                torch.zeros_like(distance)).type(torch.int8)

            confusion_matrix += create_confusion_matrix(true_labels, predicted_labels)

            # calculate loss
            loss = loss_fn(true_labels, distance).sum()

            loss_sum += loss.item()
            num_rows += true_labels.size(0)

    accuracy, precission, recall, specificity, f1, tp, fn, fp, tn = get_confusion_matrix_stats(confusion_matrix)

    return {
        "loss": loss_sum,
        "avg_loss": loss_sum / num_rows,
        "accuracy": accuracy,
        "precission": precission,
        "recall": recall,
        "specificity": specificity,
        "F1": f1,
        "tp": tp,
        "fn": fn,
        "fp":fp,
        "tn": tn,
    }

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])        
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
    ])
    
    return train_transforms, val_transforms

def save_model(name, model):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format
    model_scriped = torch.jit.script(model)
    model_scriped.save(config["out_dir"]  / f"{name}_model.pt")

def save_confusion_matrix(confusion_matrix, path):
    df_cm = pd.DataFrame(confusion_matrix, 
                         index = ["True", "False"],
                      columns = ["True", "False"],
                      )
    ax = sn.heatmap(df_cm, annot=True, fmt='.0f', square=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')

def print_metrics(metrics):
    for k,v in metrics.items():
        print(f"- {k}: {v}")

def save_statistics_csv(statistics):
    """
    Save training and validation statistics to csv file
    """
    csv = {
        "Epoch": [],
        "Learning rate": []
    }
    for i in range(len(statistics["epoch"])):
        csv["Epoch"].append(i+1)
        csv["Learning rate"].append(statistics["lr"][i])

        for k,v in statistics["train"][i].items():
            key = f"Training {k}"
            if key not in csv:
                csv[key] = []
            csv[key].append(v)

        for k,v in statistics["val"][i].items():
            key = f"Validation {k}"
            if key not in csv:
                csv[key] = []
            csv[key].append(v)

    pd.DataFrame(csv).to_csv(config["out_dir"] / "statistics.csv", index = False)
    
def train_model():
    os.makedirs(config["out_dir"])

    val_images = ImageFolder(config["dataset_dir"] / "validation")
    train_images = ImageFolder(config["dataset_dir"] / "train")
    
    train_transform, val_transform = get_transforms()
    
    dataset_val = FishPairsDataset(
        fish_dataset = val_images,
        transform = val_transform, 
        pairs = create_all_possible_pairs(len(val_images))
        #pairs = create_pairs(len(val_images), 5)
    )
    
    dataset_train = TwoRandomFishesDataset(
        fish_dataset = train_images, 
        transform = train_transform,
        # The dataloader will pick -size- number of images each epoch
        size = len(train_images) * 5, 
    )
    
    train_dataloader = DataLoader(
        dataset_train,
        batch_size = config["batch_size"], 
        num_workers = 4, 
        prefetch_factor = 2, 
        persistent_workers = True
    )
    
    val_dataloader = DataLoader(
        dataset_val,
        batch_size = 64,
        num_workers = 4, 
        prefetch_factor = 2, 
        persistent_workers = True, 
        shuffle = False
    )

    device = config["device"]
    
    #model = VGG11Siamese(feature_vector_size = 32).to(device)
    model = CustomSiameseNetwork(feature_vector_size = 32).to(device)
    
    loss_fn =  ContrastiveLoss(
        alpha = config["CL_alpha"], 
        beta = config["CL_beta"], 
        margin = config["CL_margin"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr = config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        config["lrs_step_size"],  
        config["lrs_gamma"]
    )

    statistics = {
        "train": [],
        "val": [],
        "epoch": [],
        "lr": []
    }
    
    lowest_loss = float('inf')
    highest_accuracy = 0
    highest_f1 = 0
    epoch = 1
    done = False

    while not done:
        print(f"\n*** Epoch {epoch} ***")
        
        # Training
        train_metrics = train(model, train_dataloader, optimizer, loss_fn)
        statistics["train"].append(train_metrics)
        print("Training")
        print_metrics(train_metrics)

        # Validation
        val_metrics = validate(model, val_dataloader, loss_fn)
        statistics["val"].append(val_metrics)
        print("Validation")
        print_metrics(val_metrics)

        if val_metrics["loss"] < lowest_loss:
            print("New loss record. Saving model..")
            save_model(f"lowest_loss", model)
            lowest_loss = val_metrics["loss"]
            
        if val_metrics["F1"] > highest_f1:
            print("New f1 record. Saving model..")
            save_model(f"highest_f1", model)
            highest_f1 = val_metrics["F1"]
        
        if val_metrics["accuracy"] > highest_accuracy:
            print("New accuracy record. Saving model..")
            save_model(f"highest_accuracy", model)
            highest_accuracy = val_metrics["accuracy"]
        
        lr_scheduler.step()
        statistics["epoch"].append(epoch)
        statistics["lr"].append(lr_scheduler.get_last_lr()[0])
        
        save_statistics_csv(statistics)

        done = lr_scheduler.get_last_lr()[0] <= config["stop_lr"]
        epoch += 1

def test_model(model):
    os.makedirs(config["testing_dir"], exist_ok=True)
    
    test_images = ImageFolder(config["dataset_dir"] / "test")
    
    _, test_transform = get_transforms()
    
    device = config["device"]

    all_possible_pairs = create_all_possible_pairs(len(test_images))
    shuffle(all_possible_pairs)

    dataset_test = FishPairsDataset(
        fish_dataset = test_images,
        transform = test_transform, 
        pairs = all_possible_pairs
    )

    test_dataloader = DataLoader(
        dataset_test,
        batch_size = 32,
        num_workers = 4,
        prefetch_factor = 2,
        persistent_workers = True, 
        shuffle = False
    )

    confusion_matrix = np.zeros((2, 2))
    distances_false_label = [] # Distances between fishes of different species
    distances_true_label = [] # Distances between fishes of same species
    i = 0
    for batch in tqdm(test_dataloader, desc = "Testing"):
        imgs1, imgs2, true_labels = batch
        
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        true_labels = true_labels.to(device)
        
        distance = model(imgs1, imgs2)
        
        predicted_labels = torch.where(
            distance < 0.5, 
            torch.ones_like(distance), 
            torch.zeros_like(distance)).type(torch.int8) 

        distances_true_label += distance[torch.where(true_labels == 1)].tolist()
        distances_false_label += distance[torch.where(true_labels == 0)].tolist()

        confusion_matrix += create_confusion_matrix(true_labels, predicted_labels)

    accuracy, precission, recall, specificity, f1, tp, fn, fp, tn = get_confusion_matrix_stats(confusion_matrix)

    d0mean = np.mean(distances_false_label)
    d1mean = np.mean(distances_true_label)
    d0std = np.std(distances_false_label)
    d1std = np.std(distances_true_label)
    
    save_confusion_matrix(confusion_matrix, config["testing_dir"] / "confusion.png")
    pd.DataFrame({
        "Accuracy": [accuracy],
        "Precission": [precission],
        "Recall": [recall],
        "Specificity": [specificity],
        "F1": [f1],
        "tp": [tp],
        "fn": [fn],
        "fp":[fp],
        "tn": [tn],
        "d0mean": [d0mean],
        "d1mean":[d1mean],
        "d0std":[d0std],
        "d1std":[d1std]
    }).to_csv(config["testing_dir"]  / "stats.csv", index = False)
    print("Saved statistics to ", config["testing_dir"]  / "stats.csv")

    plt.hist([distances_true_label, distances_false_label], color=["green", "red"], label=["Same species", "Not same"], stacked=True)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(config["testing_dir"]  / "histogram.png")

    print("Saved histogram to ", config["testing_dir"]  / "histogram.png")

    plt.close('all')

def test_create_some_image_comparisons(model):
    os.makedirs(config["testing_dir"], exist_ok = True)
        
    test_images = ImageFolder(config["dataset_dir"] / "test")
    _, test_transform = get_transforms()

    all_pairs = create_all_possible_pairs(len(test_images))
    shuffle(all_pairs)


    # Search for n correctly predicted images, and n wrongly predicted
    image_limit_each_class = 20
    correct_counter = 0
    incorrect_counter = 0
    for idx1, idx2 in tqdm(all_pairs, desc = "Running model on some test pairs"):
        
        if incorrect_counter > image_limit_each_class and correct_counter > image_limit_each_class:
            break

        img1,l1 = test_images.__getitem__(idx1)
        img2,l2 = test_images.__getitem__(idx2)

        imgs1 = torch.stack([test_transform(img1)]).to(config["device"])
        imgs2 = torch.stack([test_transform(img2)]).to(config["device"])

        distance = model(imgs1, imgs2)[0].item()

        if distance < 0.5:
            correct = l1 == l2
        else:
            correct = l1 != l2
        

        if correct:
            correct_counter += 1
            if correct_counter > image_limit_each_class:
                continue
        else:
            incorrect_counter +=1 
            if incorrect_counter > image_limit_each_class:
                continue

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        pred = "Not the same species" if distance > 0.5 else "Same species"
        fig.suptitle(f"Distance: {round(distance, 3)}", fontsize = 'large')
        axs[0].imshow(img1)
        axs[0].set_title(f"{fish_class_to_latin_name[test_images.classes[l1]]} ({l1})")
        axs[0].axis('off')

        axs[1].imshow(img2)
        axs[1].set_title(f"{fish_class_to_latin_name[test_images.classes[l2]]} ({l2})")
        axs[1].axis('off')
        plt.tight_layout()
        #plt.show()

        ok_or_wrong = "ok" if correct else "wrong"
        file_name = config["testing_dir"] / f"{idx1}_{idx2}_{ok_or_wrong}.png"
        plt.savefig(file_name)
        print("Saved to ", file_name)
        plt.close('all')
    
def print_num_params(model):
    total = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", total)
    
if __name__ == '__main__':
    #train_model()
    #exit(0)
    with torch.no_grad():
        model = torch.jit.load(Path(__file__).parent / "lowest_loss_model_experimental.pt").to(config["device"])
        model.eval()
        #test_model(model)
        test_create_some_image_comparisons(model)