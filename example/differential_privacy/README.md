## Usage

Moment Accountant
```python
ga = GeneralMomentAccountant(noise_type="Gaussian",
                             search="greedy",
                             precision=0.001,
                             orders=list(range(2, 64)),
                             bound_type="rdp_tight_upperbound")
ga.add_step_info({"sigma":noise_multiplier}, sampling_rate, iterations)
ga.get_epsilon(delta)
```

DPSGD
```python
privacy_manager = PrivacyManager(
        accountant,
        optim.SGD,
        l2_norm_clip=l2_norm_clip,
        dataset=trainset,
        lot_size=lot_size,
        batch_size=batch_size,
        iterations=iterations,
    )

dpoptimizer_cls, lot_loader, batch_loader = privacy_manager.privatize(
        noise_multiplier=sigma
    )

for data in lot_loader(trainset):
    X_lot, y_lot = data
    optimizer.zero_grad()
    for X_batch, y_batch in batch_loader(TensorDataset(X_lot, y_lot)):
        optimizer.zero_grad_keep_accum_grads()
        pred = net(X_batch)
        loss = criterion(pred, y_batch.to(torch.int64))
        loss.backward()
        optimizer.update_accum_grads()
    optimizer.step()
```


## Comparison with Opacus

|                               | `get_epsilon`     | `get_noise_multiplier` |
| ----------------------------- | ----------------- | ---------------------- |
| Opacus                        | 19.3 ms ± 3.47 ms | 208 ms ± 14.6 ms       |
| AIJack with SGM               | 816 µs ± 180 µs   | 8.25 ms ± 1.18 ms      |
| AIJack with tight upper bound | 1.33 ms ± 310 µs  | 15.7 ms ± 7.24 ms      |
