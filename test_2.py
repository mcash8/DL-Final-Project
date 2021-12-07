import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math

#import urllib 
import urllib.request
import pickle

'''
CSC 7343 Deep Learning Final Project: Martha Cash and Emmanual Akoja
Model is adapted from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''

filename = sys.argv[1]

Train = True if filename == 'data_train.txt' else False

def openFile(file): 
    '''
    Description: Read in text file from CLI and return the input string and label in seperate arrays
    Input: file 
    Output: Two arrays, input and target, containing the input and target strings respectively 
    '''
    input = []
    output = []
    with open(file, 'r') as f:
        for line in f: 
            if ',' in line:
                currentLine = line.split(',')
                input.append(currentLine[0])
                output.append(currentLine[1].strip('\n'))
            else:
                input.append(line.strip('\n'))
    return input, output

if Train:
    input, output = openFile(filename)
    max_length = len(max(output, key=len))
else:
    predictInput, _ = openFile(filename)
    max_length = 24
    

sos_tkn = 0
eos_tkn = 1

class Sequence:
    '''
    Class to help make a dictionary. Each word from the input or target is split into letters
    and then added to the dictionary. Each letter is added to an index to keep track of frequency of letter. 
    '''
    def __init__(self): 
        self.letter2index = {}
        self.letter2count = {}
        self.index2letter = {0: "SOS", 1: "EOS"}
        self.n_letters = 2
    
    def addWord(self, word): 
         '''
         Description: split a word into letters and pass to addLetter function
         Input: word from input or target sequence
         Output: none 
         '''
         for letter in word:
            self.addLetter(letter)
     
    def addLetter(self, letter): 
        '''
        Description: Create a unique index for a letter if letter has not alread been index
        otherwise add occurance of let
        '''
        if letter not in self.letter2index:
             self.letter2index[letter] = self.n_letters
             self.letter2count[letter] = 1
             self.index2letter[self.n_letters] = letter
             self.n_letters += 1
        else:
             self.letter2count[letter] += 1

def pairData(input, output): 

    '''
    Description: Pass input and output sequence to Sequence class. Create pairs of sequences. 
    '''

    source = Sequence()
    target = Sequence()

    pairs = []

    for i in range(len(input)):
        full = [input[i], output[i]]
        source.addWord(input[i])
        target.addWord(output[i])

        pairs.append(full)

    return source, target, pairs

def loadPredict(input):
    '''
    Description: Same as pairData function but used for data_predict.txt
    '''
    predict = Sequence()

    for i in range(len(input)):
        predict.addWord(input[i])
    
    return predict

if Train:
    source, target, pairs = pairData(input, output)
else:
    predict = loadPredict(predictInput)

def indexesFromSequence(Sequence, word):
    '''
    Get indexes of letters in sequences
    '''
    return [Sequence.letter2index[letter] for letter in word]

def tensorFromSequence(Sequence, word):
    '''
    Convert sequnece to indexes and cast to tensor
    '''
    indexes = indexesFromSequence(Sequence, word)
    indexes.append(eos_tkn)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1,1)

def tensorsFromPair(pair):
    '''
    Convert pair to a tensor
    '''
    input_tensor = tensorFromSequence(source, pair[0])
    target_tensor = tensorFromSequence(target, pair[1])
    return (input_tensor, target_tensor)

def tensorsForPredict(input):
    '''
    Convert predicted sequence to a tensor
    ''' 
    predict_tensor = tensorFromSequence(predict, predictInput)
    return predict_tensor

'''
The Model
'''
class Encoder(nn.Module): 
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded 
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self): 
        return torch.zeros(1,1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_length):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_length):
    '''
    Helper function for trainIters. Perform one iteration of training. Generate a random number, if the number is larger then the teacher
    forcing ratio, then don't use teacher forcing. 
    '''
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[sos_tkn]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == eos_tkn:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


'''
Helper functions for plotting time training took. 
'''

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    return asMinutes(now - since)

def trainIters(encoder, decoder, n_iters, pairs, learning_rate=0.01):
    '''
    Train model. Print losses and time trianing took for each epoch. Plot losses every epoch. 
    '''
    start = time.time() #for tracking training time
    plot_losses = []
    print_loss_total = 0 
    plot_loss_total = 0  

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    #generate pairs and cast to tensor
    pairs = [tensorsFromPair(pair) for pair in pairs]

    #epoch
    for i in range(n_iters//len(pairs)):
        #shuffle pairs
        np.random.shuffle(pairs)
        for k in range(len(pairs)):
            #iterate through all pairs
            training_pair = pairs[k]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

      
        print_loss_avg = print_loss_total/len(pairs)
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start), (k+1), (k+1) / n_iters * 100, print_loss_avg))

        plot_loss_avg = plot_loss_total/len(pairs)
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        Plot(plot_losses)



def Plot(points):
    '''
    Function for plotting loss over an epoch
    '''
    plt.figure()
    plt.plot(points)
    plt.savefig('loss.png')
    #clear
    plt.close()
    plt.cla()
    plt.clf()

'''
Evaluate 
'''

def evalOnPred(encoder, decoder, word, max_length=max_length):
    '''
    Helper function for evaluatePredictionData. 
    '''
    with torch.no_grad(): 
        input_tensor = tensorFromSequence(predict, word)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                        encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[sos_tkn]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            _, topi = decoder_output.data.topk(1)
            if topi.item() == eos_tkn:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(target.index2letter[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluatePredictionData(encoder, decoder):
    '''
    Generate and write predictions from data_predict.txt to output file 
    '''
    filename = 'result_predict.txt'
    with open(filename, 'w') as f:
        for i in range(len(predictInput)):
            output_letters, _ = evalOnPred(encoder, decoder, predictInput[i])
            output_word = ''.join(output_letters[0:-1])
            f.write(predictInput[i])
            f.write(',')
            f.write(output_word)
            f.write('\n')

hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder1 = Encoder(22, hidden_size).to(device)
decoder = Decoder(hidden_size, 22, dropout_p=0.1).to(device)

if Train:
    trainIters(encoder1, decoder, 902224 , pairs)
    predict = loadPredict(predictInput)
else:
    #need to load target object for evalOnPred function
    urllib.request.urlretrieve('https://drive.google.com/u/0/uc?id=1dJmmnPYyvMmWQENO9hIx2KQ_ELSh51Gv&export=download', 'target.obj')
    f = open('target.obj', 'rb')
    target = pickle.load(f)
    f.close()

    #load trained models from google drive
    print('downloading trained models....')
    urllib.request.urlretrieve('https://drive.google.com/u/0/uc?id=1c39Nc1dmOvxYDDrJRjdLzjZSvxJRgF_5&export=download', 'encoder.pt')
    encoder1.load_state_dict(torch.load('encoder.pt', map_location=device))
    encoder1.eval()

    urllib.request.urlretrieve('https://drive.google.com/u/0/uc?id=1PGbmccH2JV1wKUoDkBnMv42IMjFVlA2T&export=download', 'decoder.pt')
    decoder.load_state_dict(torch.load('decoder.pt', map_location=device))
    decoder.eval()

    print('models downloaded, generating predictions...')
    evaluatePredictionData(encoder1, decoder)

    print('generated predictions!')