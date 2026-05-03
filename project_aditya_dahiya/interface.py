from config import batchsize as the_batch_size
from config import epochs as total_epochs
from dataset import TargetDataset as TheDataset
from dataset import create_dataloader as the_dataloader
from model import TargetDetector as TheModel
from predict import the_predictor
from train import the_trainer