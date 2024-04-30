import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from data import ImageDataset
from models.modules.normalization import *
from engines import server_test_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--dataset", type = str, default = "PACS"), parser.add_argument("--subdataset", type = int)
    parser.add_argument("--num_classes", type = int, default = 7)
    parser.add_argument("--num_clients", type = int, default = 3)
    parser.add_argument("--num_rounds", type = int, default = 100)
    parser.add_argument("--num_epochs", type = int, default = 1)
    args = parser.parse_args()

    server_model = torchvision.models.resnet50(norm_layer = eXplicitOptimizedNorm2d)
    server_model.load_state_dict(
        torch.load(
            "../models/hub/resnet50_xon4.pth", 
            map_location = "cpu", 
        ), 
        strict = False, 
    )
    server_model.fc = nn.Linear(
        server_model.fc.in_features, args.num_classes, 
    )
    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items() if "batchnorm" not in key]
    initial_parameters = flwr.common.ndarrays_to_parameters(initial_parameters)
    save_ckp_dir = "../../ckps/{}/{}".format(args.dataset, args.subdataset)
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)
    flwr.server.start_server(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        config = flwr.server.ServerConfig(num_rounds = args.num_rounds), 
        strategy = FedAvg(
            min_available_clients = args.num_clients, min_fit_clients = args.num_clients, 
            server_model = server_model, 
            initial_parameters = initial_parameters, 
            save_ckp_dir = save_ckp_dir, 
        ), 
    )

    test_loaders = {
        "test":torch.utils.data.DataLoader(
            ImageDataset(
                data_dir = "../../datasets/{}/{}/test/".format(args.dataset, args.subdataset), 
                augment = False, 
            ), 
            num_workers = 0, batch_size = 32, 
            shuffle = False, 
        ), 
    }
    server_model = torch.load(
        "{}/server-best.ptl".format(save_ckp_dir), 
        map_location = "cpu", 
    )
    results = server_test_fn(
        test_loaders, 
        server_model, 
        device = torch.device("cuda"), 
    )