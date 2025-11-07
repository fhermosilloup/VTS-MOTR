from datasets import ua_detrac  # esto registra tu dataset
from datasets import build_dataset
from torch.utils.data import DataLoader
import argparse

def main():
    parser = argparse.ArgumentParser()
    # si necesitas args, agrégalos
    parser.add_argument('--dataset_file', default='ua_detrac')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=0, type=int)  # para Windows conviene 0
    args = parser.parse_args()

    dataset_train = build_dataset(image_set='train', args=args)
    collate_fn = None  # ajusta si tu dataset necesita collate_fn
    data_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             collate_fn=collate_fn)
    
    # probar un batch
    for batch in data_loader:
        print(batch)
        break

if __name__ == "__main__":
    main()
