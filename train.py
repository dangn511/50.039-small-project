import argparse
import pickle
from model_utils import load_data, read_json
from model_ic_try3 import Net, validation, make_NN, save_checkpoint, save_whole_model

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
# arch argument is irrelevant
parser.add_argument("--arch", default="vgg_adapted", help="choose model architecture")
parser.add_argument("--learning_rate", type=float, default=0.001, help="set learning rate")
# hidden_units is irrelevant
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")

args = parser.parse_args()

cat_to_name = read_json(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

model, history = make_NN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data)

with open('loss_history.pickle', 'wb') as f:
    pickle.dump(history, f)


if args.save_dir:
    save_whole_model(model, args.save_dir)