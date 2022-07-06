from libs_path import *
from libs import *
torch.cuda.set_device(3)
import time
def main():
    args = get_args_1d()
    
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(args.seed, printout=False)

    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    train_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=True,
                                   train_portion=0.5,
                                   data_path=data_path,)

    valid_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=False,
                                   valid_portion=100,
                                   data_path=data_path,)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False,
                              drop_last=False, **kwargs)


    sample = next(iter(train_loader))
    seq_len = 2**13//args.subsample
    k =7

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
    print('='*(40 + len('Data loader batch')+2))

    if is_interactive():
        u0 = sample['node']
        pos = sample['pos']
        u = sample['target']
        _, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
        axes = axes.reshape(-1)
        indexes = np.random.choice(range(4), size=4, replace=False)
        for i, ix in enumerate(indexes):
            axes[i].plot(pos[ix], u0[ix], label='input')
            axes[i].plot(pos[ix], u[ix, :, 0], label='target')
            axes[i].plot(pos[ix, 1:-1], u[ix, 1:-1, 1],
                         label='target derivative')
            axes[i].legend()

    with open(os.path.join(SRC_ROOT, 'config.yml')) as f:
        config = yaml.full_load(f)
    test_name = os.path.basename(__file__).split('.')[0]
    config = config[test_name]
    config['attn_norm'] = not args.layer_norm
    for arg in vars(args):
        if arg in config.keys():
            config[arg] = getattr(args, arg)
            
    config["seq_len"] = seq_len
    config["k"] = k
    
    get_seed(args.seed)
    torch.cuda.empty_cache()
    model = SimpleTransformer(**config)
    #model =ResNet(input_width = 1, layer_width = 64) 
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    model_name, result_name = get_model_name(model='Burgers_svd%d'%k,
                                         num_encoder_layers=config['num_encoder_layers'],
                                         n_hidden=config['n_hidden'],
                                         attention_type=config['attention_type'],
                                         layer_norm=config['layer_norm'],
                                         grid_size=int(2**13//args.subsample),
                                         )
    print(f"Saving model and result in {MODEL_PATH}/{model_name}\n")

    epochs = args.epochs
    lr = args.lr
    h = (1/2**13)*args.subsample
    tqdm_mode = 'epoch' if not args.show_batch else 'batch'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4,
                           pct_start=0.2,
                           final_div_factor=1e4,
                           steps_per_epoch=len(train_loader), epochs=epochs)

    loss_func = WeightedL2Loss(regularizer=True, h=h, gamma=args.gamma)

    metric_func = WeightedL2Loss(regularizer=False, h=h)
    
    start = time.time()
    
    result = run_train(model, loss_func, metric_func,
                       train_loader, valid_loader,
                       optimizer, scheduler,
                       train_batch=train_batch_burgers,
                       validate_epoch=validate_epoch_burgers,
                       epochs=epochs,
                       patience=None,
                       tqdm_mode=tqdm_mode,
                       model_name=model_name,
                       result_name=result_name,
                       device=device)
    a = np.zeros(1)
    a[0] = (time.time()-start)/epochs
    #print("time per epoch:", a[0])
    np.save('time per epoch svd %d %s'%(k,args.attention_type), a)
#     time = result["time"]
#     time = np.mean(time)
#     np.save("mean time svd %s k=%d "%(args.attention_type, k), time)
#     np.save("mean time original %s"%args.attention_type, time)
    
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
    model.eval()
    val_metric = validate_epoch_burgers(model, metric_func, valid_loader, device)
    print(f"\nBest model's validation metric in this run: {val_metric}")
    np.save("best metric svd %s k=%d"%(args.attention_type, k), val_metric)

    plt.figure(1)
    loss_train = result['loss_train']
    loss_val = result['loss_val']
    np.save('svd_%d_%s_tran_loss_train' %(k,args.attention_type),loss_train)
    np.save('svd_%d_%s_tran_loss_val'%(k,args.attention_type), loss_val)
    
#     np.save("burgers_original_%s_train_loss"%args.attention_type, loss_train)
#     np.save('burgers_original_%s_val_loss'%args.attention_type, loss_val)
    
    plt.semilogy(loss_train[:, 0], label='train')
    plt.semilogy(loss_val, label='valid')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
#  inference time
    sample = next(iter(valid_loader))
    dummy_node = torch.randn(sample["node"].shape, dtype=torch.float).to(device)
    dummy_pos = torch.randn(sample["pos"].shape, dtype=torch.float).to(device)
    dummy_grid = torch.randn(sample["grid"].shape, dtype=torch.float).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_node, None, dummy_pos, dummy_grid)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_node, None, dummy_pos, dummy_grid)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
#     np.save("inference time burgers original %s"%args.attention_type, mean_syn)   
    
    np.save("inference time burgers svd %s k=%d"%(args.attention_type, k), mean_syn) 
    
#     sample = next(iter(valid_loader))
#     node = sample['node']
#     pos = sample['pos']
#     grid = sample['grid']
#     u = sample['target']

#     with torch.no_grad():
#         model.eval()
#         out_dict = model(node.to(device), None,
#                         pos.to(device), grid.to(device))

#     out = out_dict['attn_weights']
#     matrix = torch.zeros(len(out),args.subsample,1,seq_len, seq_len)
#     for i in range(len(out)):
#         matrix[i] = out[i] 
#     np.save('attention',matrix)

#     _, axes = plt.subplots(nrows=args.val_batch_size, ncols=1, figsize=(20, 5*args.val_batch_size))
#     axes = axes.reshape(-1)
#     for i in range(args.val_batch_size):
#         grid = pos[i, :, 0]
#         axes[i].plot(grid, node[i, :, 0], '.', color='b', linewidth=1, label='f')
#         axes[i].plot(grid, u[i, :, 0], color='g', linewidth=2, label='u')
#         axes[i].plot(grid, preds[i, :], '--', color='r', linewidth=2, label='u_preds')
#         axes[i].legend()
# #    plt.savefig("linformer127_result_1.png")
#     plt.show()

if __name__ == '__main__':
    main()
