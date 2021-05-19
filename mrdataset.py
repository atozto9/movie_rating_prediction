from torch.utils.data import Dataset, DataLoader
import torch
import utils
import text


class MovieRateDataset(Dataset):
    def __init__(self, text_path, label_path, sort_text=False):
        self.text = utils.load_data(text_path)

        # 0~9
        if label_path == None:
            self.label = [1] * len(self.text)
        else:
            label = utils.load_data(label_path)
            self.label = [int(x.strip())-1 for x in label]

        if sort_text:
            tmp = sorted(zip(self.text, self.label), key=lambda x: len(x[0]))
            self.text, self.label = list(zip(*tmp))

        self.text_converter = text.KoreanText(symbol_from_data=False)

    def __getitem__(self, index):
        return self.text[index], self.label[index]

    def __len__(self):
        return len(self.label)

    def collate_batch(self, batch):
        label_list, text_list, len_list = [], [], []
        coarse_label_list = []

        for text, label in batch:
            label_list.append(label)
            if label >= 8:
                coarse_label_list.append(1)
            else:
                coarse_label_list.append(0)

            converted_text = self.text_converter.text_to_idx(text)
            if len(converted_text) == 0:
                converted_text = self.text_converter.text_to_idx(".")

            processed_text = torch.tensor(converted_text, dtype=torch.int64)
            text_list.append(processed_text)
            len_list.append(processed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        len_list = torch.tensor(len_list)
        text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)

        coarse_label_list = torch.tensor(coarse_label_list, dtype=torch.float32)

        return label_list, text_list, len_list, coarse_label_list


if __name__ == '__main__':
    print("dataset")

    train_dataset = MovieRateDataset(text_path='../filtered_train_data', label_path='../filtered_train_label',
                                     sort_text=True)
    print(train_dataset[0:3])

    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_batch)

    for x in train_dataloader:
        print(x)
        break
