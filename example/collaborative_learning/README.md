# Collaborative Learning

AIJack allows you to easily experiment collaborative learning such as federated learning and split learning. All you have to do is add a few lines of code to the regular pytorch code.

- federated learning

```python
clients = [TorchModule(), TorcnModule()]
global_model = TorchModule()
server = FedAvgServer(clients, global_model)

for _ in range(epoch):

  for client in clients:
    normal pytorch training.

  server.update()
  server.distribtue()
```

- split learning

```python
client_1 = SplitNNClient(first_model, user_id=0)
client_2 = SplitNNClient(second_model, user_id=1)
clients = [client_1, client_2]
splitnn = SplitNN(clients)

for _ in range(epoch):
  for x, y in dataloader:

    for opt in optimizers:
      opt.zero_grad()

    pred = splitnn(x)
    loss = criterion(y, pred)
    loss.backwad()
    splitnn.backward()

    for opt in optimizers:
      opt.step()
```
