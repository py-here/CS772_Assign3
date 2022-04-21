import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)
        
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        
        outputs = []   
        output_length = 0
        
        while (output_length != max_len+1):
            output, states = self.lstm(inputs,states)
            output = self.linear(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            if (predicted_index == 1):
                break
            
            inputs = self.embedding_layer(predicted_index)   
            inputs = inputs.unsqueeze(1)
           
            output_length += 1

        return outputs
        
        
        #Comments for the sample function
            #LSTM layer
            # input  : (1,1,embed_size)
            # output : (1,1,hidden_size)
            # States should be passed to LSTM on each iteration in order for it to recall the last word it produced.
            
            #Linear layer
            # input  : (1,hidden_size)
            # output : (1,vocab_size)
            
            # CUDA tensor has to be first converted to cpu and then to numpy.
            # Because numpy doesn't support CUDA ( GPU memory ) directly.
            # See this link for reference : https://discuss.pytorch.org/t/convert-to-numpy-cuda-variable/499
       
            # <end> has index_value = 1 in the vocabulary [ Notebook 1 ]
            # This conditional statement helps to break out of the while loop,
            # as soon as the first <end> is encountered. Length of caption maybe less than 20 at this point.
            # Prepare for net loop iteration 
            # Embed the last predicted word to be the new input of the LSTM
            # To understand this step, again look at the diagram at end of  [ Notebook 1 ]
             
            # To move to the next iteration of the while loop.