import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from data import ImageDataset
from models.modules.normalization import *
from engines import client_fit_fn

class Client(flwr.client.NumPyClient):
    def __init__(self, 
        fit_loaders, num_epochs, 
        client_model, 
        client_optim, 
        save_ckp_dir, 
    ):
        self.fit_loaders, self.num_epochs,  = fit_loaders, num_epochs, 
        self.client_model = client_model
        self.client_optim = client_optim
        self.save_ckp_dir = save_ckp_dir

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            client_optim, 
            T_max = client_optim.num_rounds, 
        )

    def get_parameters(self, 
        config, 
    ):
        parameters = [value.cpu().numpy() for key, value in self.client_model.state_dict().items()]
        return parameters

    def fit(self, 
        parameters, config, 
    ):
        keys = [key for key in self.client_model.state_dict().keys() if "batchnorm" not in key]
        self.client_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(keys, parameters)}), 
            strict = False, 
        )

        self.lr_scheduler.step()
        results = client_fit_fn(
            self.fit_loaders, self.num_epochs, 
            self.client_model, 
            self.client_optim, 
            device = torch.device("cuda"), 
        )
        torch.save(
            self.client_model, 
            "{}/client-last.ptl".format(self.save_ckp_dir), 
        )

        return self.get_parameters({}), len(self.fit_loaders["fit"].dataset), results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--dataset", type = str, default = "PACS"), parser.add_argument("--subdataset", type = int)
    parser.add_argument("--num_classes", type = int, default = 7)
    parser.add_argument("--num_clients", type = int, default = 3)
    parser.add_argument("--num_rounds", type = int, default = 100)
    parser.add_argument("--num_epochs", type = int, default = 1)
    args = parser.parse_args()

    fit_loaders = {
        "fit":torch.utils.data.DataLoader(
            ImageDataset(
                data_dir = "../../datasets/{}/{}/fit/".format(args.dataset, args.subdataset), 
                augment = True, 
            ), 
            num_workers = 0, batch_size = 32, 
            shuffle = True, 
        ), 
        "evaluate":torch.utils.data.DataLoader(
            ImageDataset(
                data_dir = "../../datasets/{}/{}/evaluate/".format(args.dataset, args.subdataset), 
                augment = False, 
            ), 
            num_workers = 0, batch_size = 32, 
            shuffle = False, 
        ), 
    }
    client_model = torchvision.models.resnet50(norm_layer = eXplicitOptimizedNorm2d)
    client_model.load_state_dict(
        torch.load(
            "../models/hub/resnet50_xon4.pth", 
            map_location = "cpu", 
        ), 
        strict = False, 
    )
    client_model.fc = nn.Linear(
        client_model.fc.in_features, args.num_classes, 
    )
    client_optim = optim.SGD(
        client_model.parameters(), weight_decay = 5e-4, 
        lr = 0.002, momentum = 0.9, 
    )
    client_optim.num_rounds = args.num_rounds

    save_ckp_dir = "../../ckps/{}/{}".format(args.dataset, args.subdataset)
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)
    client = Client(
        fit_loaders, args.num_epochs, 
        client_model, 
        client_optim, 
        save_ckp_dir, 
    )
    flwr.client.start_numpy_client(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        client = client, 
    )