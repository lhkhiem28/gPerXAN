import os, sys
from libs import *

def client_fit_fn(
    fit_loaders, num_epochs, 
    client_model, 
    client_optim, 
    device = torch.device("cpu"), 
):
    print("\nStart Client Fitting ...\n" + " = "*16)
    client_model = client_model.to(device)
    server_model_fc = copy.deepcopy(client_model.fc)
    for parameter in server_model_fc.parameters():
        parameter.requires_grad = False

    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)
        client_model.train()
        for images, labels in tqdm.tqdm(fit_loaders["fit"]):
            images, labels = images.to(device), labels.to(device)

            features = client_model.backbone(images.float())
            loss = F.cross_entropy(client_model.fc(features), labels) + 0.5*F.cross_entropy(server_model_fc(features), labels)

            loss.backward()
            client_optim.step(), client_optim.zero_grad()

    with torch.no_grad():
        client_model.eval()
        running_loss, running_corrects = 0.0, 0.0
        for images, labels in tqdm.tqdm(fit_loaders["evaluate"]):
            images, labels = images.to(device), labels.to(device)

            logits = client_model(images.float())
            loss = F.cross_entropy(logits, labels)

            running_loss, running_corrects = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item()
    evaluate_loss, evaluate_accuracy = running_loss/len(fit_loaders["evaluate"].dataset), running_corrects/len(fit_loaders["evaluate"].dataset)
    print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format("evaluate", 
        evaluate_loss, evaluate_accuracy
    ))

    print("\nFinish Client Fitting ...\n" + " = "*16)
    return {
        "evaluate_loss":evaluate_loss, "evaluate_accuracy":evaluate_accuracy
    }

def server_test_fn(
    test_loaders, 
    server_model, 
    device = torch.device("cpu"), 
):
    print("\nStart Server Testing ...\n" + " = "*16)
    server_model = server_model.to(device)

    with torch.no_grad():
        server_model.eval()
        running_loss, running_corrects = 0.0, 0.0
        for images, labels in tqdm.tqdm(test_loaders["test"]):
            images, labels = images.to(device), labels.to(device)

            logits = server_model(images.float())
            loss = F.cross_entropy(logits, labels)

            running_loss, running_corrects = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item()
    test_loss, test_accuracy = running_loss/len(test_loaders["test"].dataset), running_corrects/len(test_loaders["test"].dataset)
    print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format("test", 
        test_loss, test_accuracy
    ))

    print("\nFinish Server Testing ...\n" + " = "*16)
    return {
        "test_loss":test_loss, "test_accuracy":test_accuracy
    }