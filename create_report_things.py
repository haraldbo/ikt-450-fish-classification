import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn

file_dir = Path(__file__).parent 
train_dir = file_dir / "statistics" / "training"
test_dir = file_dir / "statistics" / "testing"

experimental_training = pd.read_csv(train_dir / "experimental" / "train_statistics_experimental.csv")
vgg_training = pd.read_csv(train_dir / "vgg11" / "train_statistics_vgg11.csv")

experimental_testing = pd.read_csv(test_dir / "experimental" / "stats.csv")
vgg_testing = pd.read_csv(test_dir / "vgg11" / "stats.csv")

def create_training_loss_graphs():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    
    axes[0].plot(vgg_training["Training avg_loss"], label = "Training")
    axes[0].plot(vgg_training["Validation avg_loss"], label = "Validation", color = "orange")
    lowest_loss_idx = vgg_training["Validation avg_loss"].idxmin()
    axes[0].scatter(
        [lowest_loss_idx], 
        vgg_training["Validation avg_loss"][lowest_loss_idx], 
        label = f"{round(vgg_training["Validation avg_loss"][lowest_loss_idx], 3)}", 
        color = "orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Avg. loss")
    axes[0].set_title("VGG11")
    axes[0].legend()

    axes[1].plot(experimental_training["Training avg_loss"], label = "Training")
    axes[1].plot(experimental_training["Validation avg_loss"], label = "Validation", color = "orange")
    lowest_loss_idx = experimental_training["Validation avg_loss"].idxmin()
    axes[1].scatter(
        [lowest_loss_idx], 
        experimental_training["Validation avg_loss"][lowest_loss_idx], 
        label = f"{round(experimental_training["Validation avg_loss"][lowest_loss_idx], 3)}", 
        color = "orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Avg. loss")
    axes[1].set_title("Experimental")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(file_dir / "loss.png")
    plt.show()

def create_validation_accuracy_f1_graphs():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    
    
    axes[0].plot(vgg_training["Validation accuracy"], label = "Accuracy")
    high_accuracy_idx = vgg_training["Validation accuracy"].idxmax()
    axes[0].scatter(
        [high_accuracy_idx], 
        [vgg_training["Validation accuracy"][high_accuracy_idx]], 
        label = f"{round(vgg_training["Validation accuracy"][high_accuracy_idx], 3)}")
    axes[0].plot(vgg_training["Validation F1"], label = "F1")
    high_f1_idx = vgg_training["Validation F1"].idxmax()
    axes[0].scatter(
        [high_f1_idx], 
        [vgg_training["Validation F1"][high_f1_idx]], 
        label = f"{round(vgg_training["Validation F1"][high_f1_idx], 3)}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("VGG11")
    axes[0].legend()

    axes[1].plot(experimental_training["Validation accuracy"], label = "Accuracy")
    high_accuracy_idx = experimental_training["Validation accuracy"].idxmax()
    axes[1].scatter(
        [high_accuracy_idx], 
        [experimental_training["Validation accuracy"][high_accuracy_idx]], 
        label = f"{round(experimental_training["Validation accuracy"][high_accuracy_idx], 3)}")
    axes[1].plot(experimental_training["Validation F1"], label = "F1")
    high_f1_idx = experimental_training["Validation F1"].idxmax()
    axes[1].scatter(
        [high_f1_idx], 
        [experimental_training["Validation F1"][high_f1_idx]], 
        label = f"{round(experimental_training["Validation F1"][high_f1_idx], 3)}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Experimental")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(file_dir / "acc_f1.png")
    plt.show()


def create_confusion_matrices():
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        def create_confusion_matrix(csv):
            conf =  np.array([
                [csv["tp"][0], csv["fn"][0]],
                [csv["fp"][0], csv["tn"][0]]
            ])

            return pd.DataFrame(conf, index = ["True", "False"], columns = ["True", "False"])

        
        confusion_matrix_vgg = create_confusion_matrix(vgg_testing)
        ax = sn.heatmap(confusion_matrix_vgg, annot=True, fmt='.0f', square=True, ax = axes[0], cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("VGG11")

        confusion_matrix_experimental = create_confusion_matrix(experimental_testing)
        ax = sn.heatmap(confusion_matrix_experimental, annot=True, fmt='.0f', square=True, ax = axes[1], cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Experimental")

        plt.savefig(file_dir / "confusion.png", bbox_inches='tight')
        plt.close('all')


create_training_loss_graphs()
#create_validation_accuracy_f1_graphs()
#create_confusion_matrices()