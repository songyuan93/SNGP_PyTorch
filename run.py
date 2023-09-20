from dataloader import *
from train import train, test_plot
from sngp import SNGP


model = SNGP()
print(model)

# #load two moon data
train_examples, train_labels, test_data, ood_data = two_moon_data(show_plot=False)

torch.nn.init.normal_(model.input_layer.weight, mean=0.0, std=0.1)
torch.nn.init.normal_(model.gp_layer.gp_output_layer.weight, mean=0.0, std=0.1)

trainer = train(model=model, epochs=100, lr=1e-4)
model = trainer.train(train_examples, train_labels)

tester = test_plot(model=model,test_data=test_data)
sngp_prob = tester.test(plot_surface=False)

plot_predictions(sngp_prob,filename="sngp_torch.png")
