# pytorch_trainer
Tips and tricks adapted from fastai+others to use for pytorch training


### LR Finder + older implemenation of one cycle policy
https://github.com/nachiket273/One_Cycle_Policy

### AdamW + Warmup schedules
https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW

- uses number of steps so need to adjust for n_epochs
Something like

### Saving best model
https://discuss.pytorch.org/t/how-to-save-a-model-from-a-previous-epoch/20252/4

### Training loop
```
def fit_fc(optimizer, num_training_steps, pct_flat=0.72, num_cycles=0.5, last_epoch=-1): # for RAdam/Ranger
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    num_warmup_steps = math.round(num_training_steps*pct_flat)
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return 1.0
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(train_dataset, model, n_epochs, bs):
    total_steps = n_epochs*math.ceil(len(train_dataset)/bs)
    best_val_accuracy = 0
    # use lr schedule here
    model.train()
    for i in tqdm(range(n_epochs)):
        for x, y in train_dataloader:
            ...
  
        for x, y in val_dataloader:
            pred = model(x)
            val_accuracy = accuracy(pred, y)
            if val_accuracy >= best_val_accuracy:
                print(f"better model found at epoch {} with val_accuracy {}")
                model.save('save_model.pkl')
        
    print(f'epoch {}: loss: {} val_loss {} val_accuracy{}')
```
