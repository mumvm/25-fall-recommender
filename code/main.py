import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

metrics_log_path = join(
    world.CODE_PATH,
    f"logs_{world.dataset}_{world.config['optimizer']}_{world.model_name}-{time.strftime('%m-%d-%Hh%Mm%Ss')}.csv"
)
metrics_logger = utils.MetricsRecorder(metrics_log_path, world.topks)
print(f"logging metrics to {metrics_log_path}")

try:
    for epoch in range(world.TRAIN_epochs):
        train_loss, output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        should_eval = (epoch % world.config.get('test_interval', 1) == 0)
        if should_eval:
            cprint("[TEST]")
            epoch_metrics = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            metrics_logger.log(epoch + 1, epoch_metrics, train_loss=train_loss)
        else:
            metrics_logger.log(epoch + 1, {}, train_loss=train_loss)
        torch.save(Recmodel.state_dict(), weight_file)
        # torch.save(Recmodel, weight_file)
finally:
    if world.tensorboard:
        w.close()
