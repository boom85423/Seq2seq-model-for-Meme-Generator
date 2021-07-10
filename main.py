from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import skimage.io as io
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import random
from collections import OrderedDict


class MemeDataset(Dataset):
    def __init__(self, vision_dir='template/', input_file='meme_input_train.npy', output_file='meme_output_train.npy',
                 vision_transform=None, tokenizer=None, RPN=None, bottleneck=None, args=None):
        
        self.vision_dir = vision_dir
        self.input_file = np.load(input_file, allow_pickle=True).tolist()
        self.output_file = np.load(output_file, allow_pickle=True).tolist()
        self.key = list(self.input_file.keys())
        self.vision = {'%s'%template:io.imread(self.vision_dir + "/" + template) for template in os.listdir(vision_dir)}
        self.vision_transform = vision_transform
        self.tokenizer = tokenizer
        self.input_max_length = args.input_max_length 
        self.output_max_length = args.output_max_length
        self.RPN = RPN
        self.bottleneck = bottleneck
        self.roi = torchvision.ops.RoIAlign((args.roi_height, args.roi_width), 1/args.scale_ratio, -1)
        # self.roi = torchvision.ops.RoIPool((args.roi_height, args.roi_width), 1/args.scale_ratio)
        self.max_num_boxes = args.max_num_boxes
        self.average_pooling_2d = nn.AdaptiveAvgPool2d((1,1))
        self.device = args.device

    def __len__(self):
        return len(self.output_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        meme_key = self.key[idx]
        template = "-".join(meme_key.split('-')[:-1])
        num_boxes = int(template.split('-')[-1].replace('.jpg', ''))

        image = self.vision[template]
        image = Image.fromarray(image)
        image = self.vision_transform(image)
        image = image.to(self.device)
        predictions = self.RPN([image])[0]
        boxes = predictions['boxes'][:num_boxes]

        # weight, height = 0, 0
        # for box in boxes:
        #     weight += (box[2] - box[0])
        #     height += (box[3] - box[1])
        # weight = weight / num_boxes
        # height = height / num_boxes

        feature = self.bottleneck(image.unsqueeze(0)) 
        objects = self.roi(feature, [boxes]) # (num_boxes, dim, w, h)
        objects = self.average_pooling_2d(objects).squeeze().view(-1, args.vision_dim)
        while objects.shape[0] < self.max_num_boxes:
            objects = torch.vstack([objects, torch.zeros(1, objects.shape[1]).to(self.device)]) 
            boxes = torch.vstack([boxes, torch.zeros(1, 4).to(self.device)])

        input_sent = self.input_file[meme_key]
        input_sent = torch.tensor(self.tokenizer(input_sent)['input_ids'])
        while len(input_sent) < self.input_max_length:
            input_sent = torch.cat((input_sent, torch.tensor([0])))

        output_sent = self.output_file[meme_key]
        output_sent = torch.tensor(self.tokenizer(output_sent)['input_ids'])
        output_sent = torch.where(output_sent == 102, 101, output_sent)
        output_sent[-1] = 102
        while len(output_sent) < self.output_max_length:
            output_sent = torch.cat((output_sent, torch.tensor([0])))
        
        sample = {'image':image, 'input_sent':input_sent, 'output_sent':output_sent, 'boxes':boxes, 'objects':objects}
        return sample


class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, objects):
        output, (hidden, cell) = self.rnn(objects)
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, vocab)

    def forward(self, inputs, hidden, cell):
        inputs = inputs.unsqueeze(0)
        embedded = self.embedding(inputs)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train(train_loader, device, BERT, encoder, decoder, criterion, optimizer, teacher_forcing_ratio=0.5):
    encoder.train()
    decoder.train()
    
    epoch_loss = 0
    for data in tqdm(train_loader, desc='training'):
        image, input_sent, output_sent, boxes, objects = data['image'].to(device), data['input_sent'].to(device), data['output_sent'].to(device),data['boxes'].to(device), data['objects'].to(device)
        objects = objects.permute(1, 0, 2)
        output_sent = output_sent.permute(1, 0)
        length = output_sent.shape[0]
        batch_size = output_sent.shape[1]

        image_feature, _, _ = encoder(objects)
        input_sent_feature = BERT(input_sent).last_hidden_state.mean(dim=1)
        
        hidden = torch.cat((image_feature[-1], input_sent_feature), 1)
        hidden = torch.cat(decoder.n_layers * [hidden.unsqueeze(0)])
        cell = hidden.clone()

        optimizer.zero_grad()

        outputs = torch.zeros(length, batch_size, decoder.vocab).to(device)
        inputs = output_sent[0, :]
        for t in range(1, length):
            output, hidden, cell = decoder(inputs, hidden, cell)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inputs = output_sent[t] if teacher_force else top1
        
        outputs_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, outputs_dim)
        output_sent = output_sent[1:].reshape(-1)

        loss = criterion(outputs, output_sent)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def test(test_loader, device, BERT, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc='testing'):
            image, input_sent, output_sent, boxes, objects = data['image'].to(device), data['input_sent'].to(device), data['output_sent'].to(device), data['boxes'].to(device), data['objects'].to(device)
            objects = objects.permute(1, 0, 2)
            output_sent = output_sent.permute(1, 0)
            length = output_sent.shape[0]
            batch_size = output_sent.shape[1]

            image_feature, _, _ = encoder(objects) 
            input_sent_feature = BERT(input_sent).last_hidden_state.mean(dim=1)
            
            hidden = torch.cat((image_feature[-1], input_sent_feature), 1)
            hidden = torch.cat(decoder.n_layers * [hidden.unsqueeze(0)])
            cell = hidden.clone()

            outputs = torch.zeros(length, batch_size, decoder.vocab).to(device)
            inputs = output_sent[0, :]
            for t in range(1, length):
                output, hidden, cell = decoder(inputs, hidden, cell)
                outputs[t] = output
                top1 = output.argmax(1)
                inputs = top1

            outputs_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, outputs_dim)
            output_sent = output_sent[1:].reshape(-1)

            loss = criterion(outputs, output_sent)
            epoch_loss += loss.item()
        return epoch_loss / len(test_loader)


def demo(vocab, demo_loader, BERT, encoder, decoder, n, device):
    count = 0
    for data in demo_loader:
        if count >= n:
            break

        image, input_sent, output_sent, boxes, objects = data['image'].to(device), data['input_sent'].to(device), data['output_sent'].to(device), data['boxes'].to(device), data['objects'].to(device)
        objects = objects.permute(1, 0, 2)
        output_sent = output_sent.permute(1, 0)
        length = output_sent.shape[0]
        batch_size = output_sent.shape[1]

        image_feature, _, _ = encoder(objects) 
        input_sent_feature = BERT(input_sent).last_hidden_state.mean(dim=1)
        
        hidden = torch.cat((image_feature[-1], input_sent_feature), 1)
        hidden = torch.cat(decoder.n_layers * [hidden.unsqueeze(0)])
        cell = hidden.clone()

        outputs = torch.zeros(length, batch_size, decoder.vocab).to(device)
        inputs = output_sent[0, :]
        for t in range(1, length):
            output, hidden, cell = decoder(inputs, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            inputs = top1
        
        outputs = outputs.argmax(2)
        print('>>> truth:', " ".join([vocab[i] for i in output_sent[1:].squeeze(1)[1:]]))
        print('>>> predict:', " ".join([vocab[i] for i in outputs[1:].squeeze(1)[1:]]))
        count += 1


def config():
    global args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_max_length', default=11, type=int)
    parser.add_argument('--output_max_length', default=14, type=int)
    parser.add_argument('--max_num_boxes', default=5, type=int)
    parser.add_argument('--scale_ratio', default=32, type=int)
    parser.add_argument('--roi_width', default=4, type=int)
    parser.add_argument('--roi_height', default=5, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--vision_dim', default=512, type=int)
    parser.add_argument('--hid_dim', default=1000, type=int)
    parser.add_argument('--sent_dim', default=768, type=int)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--drop_out', default=0.2, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    template_dir = 'template/'
    max_resolution = 0
    for template in os.listdir(template_dir):
        image = io.imread(template_dir + template)
        h, w, c = image.shape
        resolution = max(h, w)
        if resolution > max_resolution:
            max_resolution = resolution
    args.max_resolution = max_resolution
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    vision_transform = transforms.Compose([
        transforms.Resize((args.max_resolution, args.max_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.vocab = len(tokenizer)

    RPN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(args.device)
    for param in RPN.parameters():
        param.requires_grad_(False)
    RPN.eval()

    ResNet = nn.Sequential(*list(torch.load('checkpoint/ResNet.t7').children())[:-2]).to(args.device)
    for param in ResNet.parameters():
        param.requires_grad_(False)
    ResNet.eval()

    BERT = BertModel.from_pretrained('bert-base-uncased').to(args.device)
    for param in BERT.parameters():
        param.requires_grad_(False)
    BERT.eval()

    train_meme = MemeDataset(vision_dir='template/', input_file='data/meme_input_train.npy', output_file='data/meme_output_train.npy',
                             vision_transform=vision_transform, tokenizer=tokenizer, RPN=RPN, bottleneck=ResNet, args=args)
    train_loader = DataLoader(train_meme, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_meme = MemeDataset(vision_dir='template/', input_file='data/meme_input_test.npy', output_file='data/meme_output_test.npy',
                             vision_transform=vision_transform, tokenizer=tokenizer, RPN=RPN, bottleneck=ResNet, args=args)
    test_loader = DataLoader(test_meme, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    demo_loader = DataLoader(test_meme, batch_size=1, shuffle=True)

    encoder = Encoder(args.vision_dim, args.hid_dim, args.layer, args.drop_out).to(args.device)
    encoder.apply(init_weights)
    decoder = Decoder(args.vocab, args.sent_dim, args.hid_dim+args.sent_dim, args.layer, args.drop_out).to(args.device)
    decoder.apply(init_weights)
    decoder.embedding = BERT.embeddings.word_embeddings
    for param in decoder.parameters():
        param.requires_grad_(True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.epoch):
        train_loss = train(train_loader, args.device, BERT, encoder, decoder, criterion, optimizer)
        print('[Epoch %d] loss=%.4f' % (epoch, train_loss))

        if epoch % 10 == 0:
            test_loss = test(test_loader, args.device, BERT, encoder, decoder, criterion)
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                state = {'encoder':encoder.state_dict(), 'decoder':decoder.state_dict()}
                torch.save(state, 'checkpoint/model.t7')
                print('Model saved !!!')
            print('test_loss=%.4f, best_loss=%.4f (epoch=%d)' % (test_loss, best_loss, best_epoch))

            demo(list(tokenizer.vocab), demo_loader, BERT, encoder, decoder, 3, args.device)
