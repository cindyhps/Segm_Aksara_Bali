import argparse
import yaml
import torch
import os

def train_model(img_size, batch_size, epochs, data_yaml, weights, name):
    # Load YAML data
    with open(data_yaml) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # Update YAML data
    data['train'] = os.path.abspath(data['train'])
    data['val'] = os.path.abspath(data['val'])
    data['nc'] = data['nc'] if 'nc' in data else 1

    # Update batch size and number of classes
    data['batch'] = batch_size
    data['nc'] = data['nc']

    # Save updated YAML data
    with open(data_yaml, 'w') as f:
        yaml.dump(data, f)

    # Train the model
    cmd = f"python train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights {weights} --name {name}"
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Training Script")
    parser.add_argument("--img_size", type=int, default=640, help="Input image size for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--data_yaml", type=str, default="aksara_bali.yaml", help="Path to the data YAML file")
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="Path to the initial weights")
    parser.add_argument("--name", type=str, default="yolo_aksara_bali", help="Name of the output directory for the trained model")
    args = parser.parse_args()

    train_model(args.img_size, args.batch_size, args.epochs, args.data_yaml, args.weights, args.name)

if __name__ == "__main__":
    main()
