# pytorch_trainer
Tips and tricks adapted from fastai+others to use for pytorch training


### LR Finder + older implemenation of one cycle policy
https://github.com/nachiket273/One_Cycle_Policy

### AdamW + Warmup schedules
https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW

- uses number of steps so need to adjust for n_epochs
Something like

```
def fit_fc(optimizer, n_steps, pct_flat=0.72, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step <= n_steps*pct_flat:
            return 1.0
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(train_dataset, model, n_epochs, bs):
  total_steps = n_epochs*math.ceil(len(train_dataset)/bs)
  # use lr schedule here
  ...
  

```
