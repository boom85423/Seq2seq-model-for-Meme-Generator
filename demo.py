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
import cv2


class MemeDataset():
    def __init__(self, image_path, image_sent,
                 vision_transform=None, tokenizer=None, RPN=None, bottleneck=None, args=None):
        
        self.vision_transform = vision_transform
        self.tokenizer = tokenizer
        self.RPN = RPN
        self.bottleneck = bottleneck
        self.roi = torchvision.ops.RoIAlign((args.roi_height, args.roi_width), 1/args.scale_ratio, -1)
        self.max_num_boxes = args.max_num_boxes
        self.average_pooling_2d = nn.AdaptiveAvgPool2d((1,1))
        self.device = args.device

    def process(self, image_path, image_sent):
        vision = io.imread(image_path)
        image = Image.fromarray(vision)
        image = self.vision_transform(image)
        image = image.to(self.device)
        predictions = self.RPN([image])[0]
        boxes = predictions['boxes'][:self.max_num_boxes] # boxes sorting by scores

        feature = self.bottleneck(image.unsqueeze(0)) 
        objects = self.roi(feature, [boxes]) # (num_boxes, dim, w, h)
        objects = self.average_pooling_2d(objects).squeeze().view(-1, args.vision_dim)
        while objects.shape[0] < self.max_num_boxes:
            objects = torch.vstack([objects, torch.zeros(1, objects.shape[1]).to(self.device)]) 
            boxes = torch.vstack([boxes, torch.zeros(1, 4).to(self.device)])
        
        input_sent = torch.tensor(self.tokenizer(image_sent)['input_ids'])
        sample = {'image_path':image_path, 'image_sent':image_sent, 'input_sent':input_sent.unsqueeze(0), 'boxes':boxes, 'objects':objects.unsqueeze(0)}
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


def splitted_putText(img, sent, boxes, box_idx, height_ratio, width_ratio):
    splitted_sent = []
    for idx,word in enumerate(sent.split()):
        splitted_sent.append(word)
        if ((idx+1)%3 == 0) and (idx != 0):
            splitted_sent.append('\n')
    sents = ' '.join(splitted_sent)

    gap = 0
    for sent in sents.split('\n'):
        sent = sent.strip()
        img = cv2.putText(img, sent, (int(boxes[box_idx,0]+10*width_ratio), int(boxes[box_idx,1]+20*height_ratio+gap*height_ratio)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5*int(np.ceil(width_ratio)), (0,0,255), int(np.ceil(width_ratio)))
        gap += 20
    return img


def demo(args, vocab, demo_loader, BERT, encoder, decoder):
    encoder.eval()
    decoder.eval()
        
    image_path, image_sent, input_sent, boxes, objects = data['image_path'], data['image_sent'], data['input_sent'].to(args.device), data['boxes'].cpu(), data['objects'].to(args.device)
    objects = objects.permute(1, 0, 2)

    image_feature, _, _ = encoder(objects)
    input_sent_feature = BERT(input_sent).last_hidden_state.mean(dim=1)

    hidden = torch.cat((image_feature[-1], input_sent_feature), 1)
    hidden = torch.cat(decoder.n_layers * [hidden.unsqueeze(0)])
    cell = hidden.clone()

    inputs = torch.tensor([101]).to(args.device)
    output_sent = ""
    while True:
        output, hidden, cell = decoder(inputs, hidden, cell)
        top1 = output.argmax(1)
        inputs = top1

        word = vocab[top1]
        if (word == '[SEP]') or (output_sent.count('[CLS]') == args.max_num_boxes-1):
            break
        output_sent += (word + ' ') 

    # reformat the output
    output_sent = output_sent.replace(" '", "")
    output_sent = ' ' + output_sent
    output_sent = output_sent.replace(' ##', '')
    output_sent = output_sent.strip()

    splitted = output_sent.split()
    single_output_sent = [splitted[0]]
    for i in range(1, len(splitted)):
        if splitted[i] != splitted[i-1]:
            single_output_sent.append(splitted[i])
    output_sent = " ".join(single_output_sent)

    boxes = np.array(boxes)
    img = cv2.imread(image_path)
    
    height, width = img.shape[0], img.shape[1]
    height_ratio = height / args.max_resolution
    width_ratio = width / args.max_resolution

    boxes[:, 0] = boxes[:, 0] * width_ratio
    boxes[:, 2] = boxes[:, 2] * width_ratio
    boxes[:, 1] = boxes[:, 1] * height_ratio
    boxes[:, 3] = boxes[:, 3] * height_ratio
    boxes = np.ceil(boxes).astype(int)

    img = cv2.rectangle(img, (boxes[0,0], boxes[0,1]), (boxes[0,2], boxes[0,3]), (0,0,255), int(np.ceil(width_ratio)))
    img = splitted_putText(img, image_sent, boxes, 0, height_ratio, width_ratio)
    output_sent = [sent for sent in output_sent.split('[CLS]') if len(sent.strip()) > 0]
    for idx,sent in enumerate(output_sent[:len(boxes)-1]):
        img = cv2.rectangle(img, (boxes[idx+1,0], boxes[idx+1,1]), (boxes[idx+1,2], boxes[idx+1,3]), (0,0,255), int(np.ceil(width_ratio)))
        img = splitted_putText(img, sent, boxes, idx+1, height_ratio, width_ratio)
    cv2.imwrite(image_path+'_meme', img)


def config():
    global args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--max_num_boxes', default=5, type=int)
    parser.add_argument('--scale_ratio', default=32, type=int)
    parser.add_argument('--roi_width', default=4, type=int)
    parser.add_argument('--roi_height', default=5, type=int)
    parser.add_argument('--vision_dim', default=512, type=int)
    parser.add_argument('--hid_dim', default=1000, type=int)
    parser.add_argument('--sent_dim', default=768, type=int)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--drop_out', default=0.2, type=float)
    parser.add_argument('--image_path', default='template', type=str)
    parser.add_argument('--image_sent', default='hello world', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.image_path == 'template'
        image_path = 'template/' + random.choice(os.listdir('template'))
    else:
        image_path = args.image_path

    if args.image_sent == 'image_sent':
        image_sent = 'Me being single 25 years and wanna have a girlfriend'
    else:
        image_sent = args.image_sent
    
    args.max_resolution = 250
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

    demo_meme = MemeDataset(image_path=image_path, image_sent=image_sent, vision_transform=vision_transform, tokenizer=tokenizer, RPN=RPN, bottleneck=ResNet, args=args)
    data = demo_meme.process(image_path, image_sent)

    encoder = Encoder(args.vision_dim, args.hid_dim, args.layer, args.drop_out).to(args.device)
    decoder = Decoder(args.vocab, args.sent_dim, args.hid_dim+args.sent_dim, args.layer, args.drop_out).to(args.device)
   
    model = torch.load('checkpoint/model.t7')
    encoder.load_state_dict(model['encoder'])
    decoder.load_state_dict(model['decoder'])
    encoder.eval()
    decoder.eval()
    
    demo(args, list(tokenizer.vocab), data, BERT, encoder, decoder)
