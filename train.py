import torch
import models
import mrdataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter('runs/bilstm_coarse')

    train_dataset = mrdataset.MovieRateDataset(text_path='../filtered_train_data', label_path='../filtered_train_label',
                                               sort_text=True)

    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=train_dataset.collate_batch)

    valid_dataset = mrdataset.MovieRateDataset(text_path='../valid_data', label_path='../valid_label')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1024, collate_fn=valid_dataset.collate_batch)

    model = models.MovieRatingModel(symbol_size=len(train_dataset.text_converter.phonemes_list)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    bce = torch.nn.BCEWithLogitsLoss()

    running_total_loss = 0.
    running_loss = 0.
    running_coars_loss = 0.

    for epoch in range(10):
        for i, (label, text, input_len, coars_label) in enumerate(train_dataloader):
            label = label.to(device)
            text = text.to(device)
            input_len = input_len.to(device)
            coars_label = coars_label.unsqueeze(1).to(device)

            optimizer.zero_grad()

            pred_label, pred_coars_label = model(text, input_len)
            loss = criterion(pred_label, label)
            coars_loss = bce(pred_coars_label, coars_label)

            total_loss = loss + coars_loss

            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()
            running_loss += loss.item()
            running_coars_loss += coars_loss.item()

            if i % 100 == 99:
                print(running_total_loss)
                global_step = epoch * len(train_dataloader) + i

                writer.add_scalar('train/total_loss', running_total_loss, global_step)
                writer.add_scalar('train/loss', running_loss, global_step)
                writer.add_scalar('train/coarse_loss', running_coars_loss, global_step)

                running_total_loss = 0.
                running_loss = 0.
                running_coars_loss = 0.

            if i % 1000 == 999:
                r_a, c_a = eval(model, valid_dataloader, device)
                print(r_a, c_a)

                global_step = epoch * len(train_dataloader) + i

                writer.add_scalar('valid/rating_acc', r_a, global_step)
                writer.add_scalar('valid/coarse_acc', c_a, global_step)

                torch.save(model.state_dict(), str(epoch) + 'epoch_bi_cnn_coa')


def eval(model, dataloader, device):
    model.eval()
    rating_acc, coarse_acc, total_count = 0, 0, 0

    coarse_checker = 0

    with torch.no_grad():
        for i, (label, text, input_len, coars_label) in enumerate(dataloader):
            label = label.to(device)
            text = text.to(device)
            input_len = input_len.to(device)
            coarse_label = coars_label.to(device)

            predicted_label, predicted_coarse_label = model(text, input_len)

            rating_acc += (predicted_label.argmax(1) == label).sum().item()

            predicted_coarse_label = torch.round(torch.sigmoid(predicted_coarse_label).squeeze())
            coarse_acc += (predicted_coarse_label == coarse_label).sum().item()

            coarse_checker += (predicted_coarse_label == 1).sum()

            total_count += label.size(0)

    print(coarse_checker.item())
    model.train()

    return rating_acc / total_count, coarse_acc / total_count


def main():
    train()


if __name__ == '__main__':

    # train_label = utils.load_data('../filtered_train_label')
    # train_label_counter = Counter(train_label)
    # weight = [train_label_counter[str(x)] / sum(train_label_counter.values()) for x in range(1, 11)]
    # w = 1 / (np.asarray(weight) * len(train_label))
    #
    # class_weights = w[[int(x) - 1 for x in train_label]]
    # print(len(train_label))
    #
    # print(w)

    # sampler = WeightedRandomSampler(class_weights, len(class_weights))


    main()
