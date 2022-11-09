# <학습자료 8-1>

# define_argparser 함수
import argparse

import torch.cuda


def define_argparser() :
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=5)

    p. add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config


def main(config) :
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)
    
    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)
    
    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        input_size = input_size,
        output_size = output_size,
        hidden_sizes = get_hidden_sizes(input_size,
                                       output_size,
                                       config.n_layers),
        use_batch_norm = not config.use_dropout,
        dropout_p = config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss
    
    if config.verbose >= 1 :
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)
    
    trainer.train(
        train_data(x[0], y[0]),
        train_data(x[0], y[0]),
    )


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
