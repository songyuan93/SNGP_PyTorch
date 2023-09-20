import math
import torch
import torch.nn as nn
from dataloader import *

class train:
    def __init__(self,model,lr = 1e-4, epochs=10) -> None:
        self.lr = lr
        self.model = model
        self.epochs = epochs

    #loss function sparse categoriacal cross entropy
    def scce(self,input=None,labels=None):
        
        m = nn.LogSoftmax(dim=1)
        criterion = nn.NLLLoss()
        loss = criterion(m(input),labels)
        return loss

    def train_one_epoch(self,train_examples, train_labels):

        self.model.gp_layer.reset_precision_matrix()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        inputs = torch.tensor(train_examples)
        labels = torch.tensor(train_labels)
        optimizer.zero_grad()
        outputs = self.model(inputs.float(), update_cov=True)
        loss = self.scce(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss

    def train(self,train_examples,train_labels):

        for epoch in range(self.epochs):
            self.model.train(True)
            
            loss = self.train_one_epoch(train_examples,train_labels)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss: {loss} sparse_categorical_accuracy: ')
        
        return self.model

class test_plot:
    def __init__(self,test_data,model) -> None:
        self.test_data = test_data
        self.model = model

    def mean_field_logits(self,logits, variances, mean_field_factor=math.pi/8):
    
        variances = torch.diagonal(variances)
        logits_scale = (1.0 + variances * mean_field_factor) ** 0.5

        if len(logits.shape) > 1:
            logits_scale = logits_scale[:, None]

        return logits/logits_scale

    def compute_posterior_mean_probability(self,logits, covmat, lambda_param=math.pi / 8.):
    # Computes uncertainty-adjusted logits using the built-in method.
        logits_adjusted = self.mean_field_logits(
            logits, covmat, mean_field_factor=lambda_param)

        m = nn.Softmax(dim=-1)
        return m(logits_adjusted)[:, 0]

    def test(self,plot_surface=None):
        test_examples = self.test_data.clone()
        self.model.train(False)
        sngp_logits, sngp_covmat = self.model(test_examples.float(), return_gp_cov=True)
        sngp_probs = self.compute_posterior_mean_probability(sngp_logits, sngp_covmat)

        if plot_surface:
            plot_predictions(sngp_probs)
        
        return sngp_probs
