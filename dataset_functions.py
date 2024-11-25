from urllib.request import urlretrieve
from pathlib import Path
import tarfile
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from random import sample, choice, shuffle
import shutil
import pandas as pd

dataset_dir = Path(__file__).parent / "dataset"

def download_dataset():
    progress_bar = tqdm(range(100), desc = "Downloading fish dataset", leave = False)
    def download_progress_hook(count, blockSize, totalSize):
        progress_bar.n = int((count * blockSize / totalSize) * 100)
        progress_bar.refresh()
    os.makedirs(dataset_dir)
    fish_tar_file = "fish.tar"
    urlretrieve(f"https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/WEBSITE/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar", dataset_dir / fish_tar_file, reporthook = download_progress_hook)
    progress_bar.desc = "Downloading fish images"
    urlretrieve(f"https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/WEBSITE/GROUNDTRUTH/RECOG/class_id.csv", dataset_dir / "labels.csv", reporthook=download_progress_hook)
    progress_bar.desc = "Downloading fish labels"
    progress_bar.close()
    print("Extracting fish tar file..")
    fish_tar = tarfile.open(dataset_dir / fish_tar_file)
    fish_tar.extractall(dataset_dir, filter="tar")
    fish_tar.close()
    print("Cleaning up temporary fish files..")
    os.remove(dataset_dir / fish_tar_file)
    print("Done!")

def get_overview_of_data():
    labels_csv = open(dataset_dir / "labels.csv")
    labels_csv.readline() # Read and ignore first row
    filename_to_class = {}
    class_to_filenames = {}
    for line in labels_csv:
        entries = line.split(",")
        fish_type = int(entries[1].strip())
        file_name = entries[0]
        filename_to_class[file_name] = fish_type
        if fish_type not in class_to_filenames.keys():
            class_to_filenames[fish_type] = [file_name]
        else:
            class_to_filenames[fish_type].append(file_name)
    return filename_to_class, class_to_filenames

def calculate_mean_rgb(dataset: Dataset):
    rgb_sum = np.zeros(3)
    for food in tqdm(dataset, "Calculating mean RGB"):
        for color in range(3):
            flattened_image = food[0][color].flatten()
            rgb_sum[color] += flattened_image.sum().item() / flattened_image.size(0)
    return rgb_sum / dataset.__len__()

def create_height_width_scatter_plot(dataset: Dataset):
    heights = []
    widths = []
    for image,_ in tqdm(dataset, desc = "Creating height-width scatter plot", leave = False):
        w,h = image.size
        widths.append(w)
        heights.append(h)
    plt.scatter(widths, heights, color = "blue")
    plt.scatter([np.mean(widths)], [np.mean(heights)], color = "red")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "hw_scatter.png")
    plt.show()

def create_dataset_for_siamese_network(max_images_each_type):    
    """
    Takes one image from each trajectory of fish_1 to fish_10 and 
    distribute them into training and validation folders.
    
    Creates an unbalanced dataset.
    """
    
    # fish types 1 to 10
    fish_types = [f"fish_{i:02d}" for i in range(1, 11)]
    
    # Get trajectories and images that belong to each trajectory
    fish = {}
    origin_dataset_path = Path(__file__).parent / "dataset" / "fish_image"
    for fish_type in fish_types:
        fish[fish_type] = {}
        for file_name in os.listdir(origin_dataset_path / fish_type):
            _, trajectory, _ = file_name.split("_")
            if "trajectory" in fish[fish_type].keys():
                fish[fish_type][trajectory].append(file_name)
            else:
                fish[fish_type][trajectory] = [file_name]

    new_dataset_path = Path(__file__).parent / "dataset" / "fish_new_2"
    train_dir = new_dataset_path / "train"
    val_dir = new_dataset_path / "validation"
    test_dir = new_dataset_path / "test"
    
    # Prevent it from getting extremely unbalanced
    
    for fish_type in fish_types:
        # 1. Divide trajectories into training, validation and testing
        trajectories = list(fish[fish_type].keys())
        shuffle(trajectories)
        
        # Cut number of images down to max_images_each_type 
        if len(trajectories) > max_images_each_type:
            trajectories = trajectories[:max_images_each_type]
            
        train_ratio = 0.8
        val_ratio = 0.1
        split_index1 = int(train_ratio * len(trajectories))
        split_index2 = split_index1 + int(val_ratio * len(trajectories))
        train = trajectories[:split_index1]
        val = trajectories[split_index1:split_index2]
        test = trajectories[split_index2:]
        
        print("Number of images for each split:")
        print("Train:", len(train))
        print("Validation:", len(val))
        print("Testing:", len(test))
        
        # 2. Distribute one image from each trajectory
        # into training, validation and testing directories
        for dir, trajs in zip([train_dir, val_dir, test_dir], 
                           [train, val, test]):
            
            os.makedirs(dir / fish_type, exist_ok = False)
            for t in trajs:
                file_name = choice(fish[fish_type][t])
                shutil.copy(origin_dataset_path / fish_type / file_name, dir / fish_type / file_name)

def print_dataset_stats():
    dataset_path = Path(__file__).parent / "dataset" / "fish_new_2"
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validation"
    test_dir = dataset_path / "test"
    
    fish_folders = [f"fish_{i:02d}" for i in range(1, 11)]
    training_images = []
    validation_images = []
    testing_images = []
    
    for fish_folder in fish_folders:
        for fish_file in os.listdir(train_dir / fish_folder):
            training_images.append(fish_file)
        
        for fish_file in os.listdir(val_dir / fish_folder):
            validation_images.append(fish_file)
        
        for fish_file in os.listdir(test_dir / fish_folder):
            testing_images.append(fish_file)
    
    print("Number of training images:", len(training_images))
    print("Number of validation images:", len(validation_images))
    print("Number of testing images:", len(testing_images))
    print("Total:", len(testing_images + training_images + validation_images))
    
    a = len(set(training_images) & set(testing_images))
    b = len(set(training_images) & set(validation_images))
    c = len(set(testing_images) & set(validation_images))
    print("Number of shared images (should be 0):", a+b+c)
    
def print_latex_table_code():
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
    dataset_path = Path(__file__).parent / "dataset" / "fish_new_2"
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validation"
    test_dir = dataset_path / "test"
    
    for i in range(0, 10):
        fish = f"fish_{i+1:02d}" 
        num_train = len(os.listdir(train_dir / fish))
        num_val = len(os.listdir(val_dir / fish))
        num_test = len(os.listdir(test_dir / fish))
        total = num_train + num_val + num_test
        print(f"{latin_names[i]} ({fish.replace("_", "\\_")}) & {num_train} & {num_val} & {num_test} & {total} \\\\")
        print("\\hline")
    
def create_dataset_csv():
    """
    Create a csv file that contains the exact way images were distributed into train/val/test
    So that it can be replicated later
    """
    dataset_path = Path(__file__).parent / "dataset" / "fish_new_2"
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validation"
    test_dir = dataset_path / "test"
    
    with open(Path(__file__).parent / "dataset.csv", "w+") as csv:
        csv.write("split,fish_type,file_name\n")
        for split in ["train", "validation", "test"]:
            for typ in os.listdir(dataset_path / split):
                for file_name in os.listdir(dataset_path / split / typ):
                    csv.write(f"{split},{typ},{file_name}\n")
    
def recreate_dataset_from_csv(file_name):
    dataset_csv = pd.read_csv(file_name)
    dataset_src = Path(__file__).parent / "dataset" / "fish_image"
    dataset_dest = Path(__file__).parent / "dataset" / "fish_new_2"
    
    # Create directories for new dataset
    print(f"Creating dataset {dataset_dest.name}")
    fish_types = dataset_csv["fish_type"].unique()
    for split in dataset_csv["split"].unique():
        for fish_type in fish_types:
            os.makedirs(dataset_dest / split / fish_type, exist_ok=False)
    
    # Populate new dataset from exising
    for _, row in tqdm(dataset_csv.iterrows(), 
                       total = len(dataset_csv.index), desc = f"Populating {dataset_dest.name} from csv {file_name}"):
        split = row["split"]
        fish_type = row["fish_type"]
        file_name = row["file_name"]
        
        src = dataset_src / fish_type / file_name
        dest = dataset_dest / split / fish_type / file_name
        shutil.copy(src, dest)
    
    print("Done!")

#download_dataset()
#create_dataset_for_siamese_network_v1()
#create_dataset_for_siamese_network_v2(max_images_each_type = 1500)
#print_latex_table_code()
#create_height_width_scatter_plot(
#    ConcatDataset([
#        ImageFolder(Path(__file__).parent / "dataset" / "fish_new_2" / "train"),
#        ImageFolder(Path(__file__).parent / "dataset" / "fish_new_2" / "validation"),
#        ImageFolder(Path(__file__).parent / "dataset" / "fish_new_2" / "test")
#    ])
#)
#print_dataset_stats()
#create_dataset_csv()



#download_dataset()
#recreate_dataset_from_csv("dataset.csv")