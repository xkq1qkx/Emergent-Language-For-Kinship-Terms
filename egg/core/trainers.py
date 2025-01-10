# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import wandb
import random

from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
import os
import pathlib
from typing import List, Optional
from tqdm import tqdm
try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
from torch.utils.data import DataLoader

from .batch import Batch
from .callbacks import (
    Callback,
    Checkpoint,
    CheckpointSaver,
    ConsoleLogger,
    TensorboardLogger,
)
from .distributed import get_preemptive_checkpoint_dir
from .interaction import Interaction
from .util import get_opts, move_to

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        test_data: DataLoader,
        val_data: DataLoader,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data: Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device

        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs

        self.update_freq = common_opts.update_freq

        if common_opts.load_from_checkpoint is not None:
            print(
                f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}"
            )
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        self.distributed_context = common_opts.distributed_context
        if self.distributed_context.is_distributed:
            print("# Distributed context: ", self.distributed_context)

        if self.distributed_context.is_leader and not any(
            isinstance(x, CheckpointSaver) for x in self.callbacks
        ):
            if common_opts.preemptable:
                assert (
                    common_opts.checkpoint_dir
                ), "checkpointing directory has to be specified"
                d = get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
                self.checkpoint_path = d
                self.load_from_latest(d)
            else:
                self.checkpoint_path = (
                    None
                    if common_opts.checkpoint_dir is None
                    else pathlib.Path(common_opts.checkpoint_dir)
                )

            if self.checkpoint_path:
                checkpointer = CheckpointSaver(
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_freq=common_opts.checkpoint_freq,
                )
                self.callbacks.append(checkpointer)

        if self.distributed_context.is_leader and common_opts.tensorboard:
            assert (
                common_opts.tensorboard_dir
            ), "tensorboard directory has to be specified"
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

        if self.distributed_context.is_distributed:
            device_id = self.distributed_context.local_rank
            torch.cuda.set_device(device_id)
            self.game.to(device_id)

            # NB: here we are doing something that is a bit shady:
            # 1/ optimizer was created outside of the Trainer instance, so we don't really know
            #    what parameters it optimizes. If it holds something what is not within the Game instance
            #    then it will not participate in distributed training
            # 2/ if optimizer only holds a subset of Game parameters, it works, but somewhat non-documentedly.
            #    In fact, optimizer would hold parameters of non-DistributedDataParallel version of the Game. The
            #    forward/backward calls, however, would happen on the DistributedDataParallel wrapper.
            #    This wrapper would sync gradients of the underlying tensors - which are the ones that optimizer
            #    holds itself.  As a result it seems to work, but only because DDP doesn't take any tensor ownership.

            self.game = torch.nn.parallel.DistributedDataParallel(
                self.game,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
            )
            self.optimizer.state = move_to(self.optimizer.state, device_id)

        else:
            self.game.to(self.device)
            # NB: some optimizers pre-allocate buffers before actually doing any steps
            # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
            # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
            self.optimizer.state = move_to(self.optimizer.state, self.device)

        if common_opts.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def eval_epoch(self,target_set):
        epoch_prediction = []
        epoch_gt = []
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.eval()  # 切换到评估模式，关闭 Dropout 等

        with torch.no_grad():  # 禁用梯度计算
            for batch_id, batch in enumerate(tqdm(target_set)):  # 使用 test_data
                if not isinstance(batch, Batch):
                    batch = Batch(batch)
                batch = batch.to(self.device)

                context = autocast() if self.scaler else nullcontext()
                with context:
                    optimized_loss, output, message = self.game(*batch)
                    message_ids = torch.argmax(message, dim=2)
                    chosen_output = torch.zeros([output.shape[0], output.shape[2]])
                    for i in range(message_ids.shape[0]):
                        for j in range(message_ids.shape[1]):
                            if message_ids[i, j] == 0:
                                chosen_output[i, :] = output[i, j-1, :]

                    output_01 = (chosen_output > 0.5).int()
                    for i in range(message_ids.shape[0]):
                        simple_labels = batch.labels.reshape(output_01.shape)[i, :batch.node_num[i]]
                        simple_output = output_01[i, :batch.node_num[i]]

                        epoch_prediction.extend(simple_output.int().numpy())
                        epoch_gt.extend(simple_labels.int().numpy())

                    # 计算平均损失，不进行梯度反向传播
                    mean_loss += optimized_loss.detach()

                n_batches += 1

        # 计算评估指标：F1 分数和准确率
        epoch_f1_score = f1_score(epoch_gt, epoch_prediction)
        epoch_acc_score = accuracy_score(epoch_gt, epoch_prediction)
        epoch_precision_score = precision_score(epoch_gt,epoch_prediction)
        epoch_recall_score = recall_score(epoch_gt,epoch_prediction)

        mean_loss /= n_batches

        return mean_loss.item(), epoch_f1_score, epoch_acc_score,epoch_precision_score,epoch_recall_score

    def train_epoch(self):
        epoch_pridiction=[]
        epoch_gt=[]
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(tqdm(self.train_data)):
            if not isinstance(batch, Batch):
                batch = Batch(batch)
            batch = batch.to(self.device)

            context = autocast() if self.scaler else nullcontext()
            with context:
                optimized_loss, output, message= self.game(*batch)
                message_ids=torch.argmax(message,dim=2)
                chosen_output=torch.zeros([output.shape[0],output.shape[2]])
                for i in range(message_ids.shape[0]):
                    for j in range(message_ids.shape[1]):
                        if message_ids[i,j]==0:
                            chosen_output[i,:]=output[i,j-1,:]
                
                output_01=(chosen_output>0.5).int()
                #print(chosen_output)
                #print(output_01)
                for i in range(message_ids.shape[0]):
                    simple_labels=batch.labels.reshape(output_01.shape)[i,:batch.node_num[i]]
                    simlple_output=output_01[i,:batch.node_num[i]]

                    epoch_pridiction.extend(simlple_output.int().numpy())
                    epoch_gt.extend(simple_labels.int().numpy())
               
                #print(output)
                # labels = batch.y
                # labels: batch_size*max_nodes
                # output: batch_size*message_max_len*max_nodes
                # you can compute loss, accuracy here
                # in their loss use all the max_len to compute loss
                # but in accuracy and eval, they use one
                # example: when max_len=5, batch_size=1, mesage is 1 2 3 4 0
                # use output[0,3,:] as the final output,(0 is stop)
                if self.update_freq > 1:
                    # throughout EGG, we minimize _mean_ loss, not sum
                    # hence, we need to account for that when aggregating grads
                    optimized_loss = optimized_loss / self.update_freq
            
            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            # if (
            #     self.distributed_context.is_distributed
            #     and self.aggregate_interaction_logs
            # ):
            #     interaction = Interaction.gather_distributed_interactions(interaction)
            # interaction = interaction.to("cpu")

            # for callback in self.callbacks:
            #     callback.on_batch_end(interaction, optimized_loss, batch_id)

            # interactions.append(interaction)
        
        epoch_f1_score=f1_score(epoch_gt,epoch_pridiction)
        epoch_acc_score = accuracy_score(epoch_gt,epoch_pridiction)
        epoch_precision_score = precision_score(epoch_gt,epoch_pridiction)
        epoch_recall_score = recall_score(epoch_gt,epoch_pridiction)
        if self.optimizer_scheduler:
            self.optimizer_scheduler.step()

        mean_loss /= n_batches
        # full_interaction = Interaction.from_iterable(interactions)
        
        return mean_loss.item(),epoch_f1_score,epoch_acc_score,epoch_precision_score,epoch_recall_score,epoch_pridiction#, full_interaction

    def train(self, n_epochs):
        all_pred_ans=[]
        wandb.init(
            project="Kinship_learning",
            config={
                "learning_rate": 0.001,
                "focal_loss_alpha": 0.75,
                "epochs": 100,
            }
        )
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)

            #train_loss, train_interaction = self.train_epoch()
            train_lossn,train_f1_scoren,train_acc_scoren ,train_precision_scoren,train_recall_scoren ,epoch_pred_ans= self.train_epoch()
            test_lossn,test_f1_scoren,test_acc_scoren,test_precision_scoren,test_recall_scoren = self.eval_epoch(self.val_data)
            all_pred_ans.append(epoch_pred_ans)
            wandb.log(
                {
                    "train_lossn": train_lossn, "train_f1_scoren": train_f1_scoren,"train_acc_scoren":train_acc_scoren,
                    "train_precision_scoren":train_precision_scoren,"train_recall_scoren":train_recall_scoren,
                    "test_lossn": test_lossn, "test_f1_scoren": test_f1_scoren,"test_acc_scoren":test_acc_scoren,
                    "test_precision_scoren":test_precision_scoren,"test_recall_scoren":test_recall_scoren,
                }
            )

            print("|  {:>4} |    {:.5f} |     {:.5f}    |     {:.5f}    |   {:.5f} |   {:.5f} |   {:.5f} |   {:.5f} |   {:.5f} |    {:.5f}    |    {:.5f}    |"
                  .format(epoch, train_lossn, train_f1_scoren, train_acc_scoren ,train_precision_scoren, train_recall_scoren, test_lossn, test_f1_scoren,  test_acc_scoren,test_precision_scoren,test_recall_scoren))


            # for callback in self.callbacks:
            #     callback.on_epoch_end(train_loss, train_interaction, epoch + 1)

            # validation_loss = validation_interaction = None
            # if (
            #     self.validation_data is not None
            #     and self.validation_freq > 0
            #     and (epoch + 1) % self.validation_freq == 0
            # ):
            #     for callback in self.callbacks:
            #         callback.on_validation_begin(epoch + 1)
            #     validation_loss, validation_interaction = self.eval()

            #     for callback in self.callbacks:
            #         callback.on_validation_end(
            #             validation_loss, validation_interaction, epoch + 1
            #         )

            # if self.should_stop:
            #     for callback in self.callbacks:
            #         callback.on_early_stopping(
            #             train_loss,
            #             train_interaction,
            #             epoch + 1,
            #             validation_loss,
            #             validation_interaction,
            #         )
            #     break

        for callback in self.callbacks:
            callback.on_train_end()
        torch.save(self.game,"../check_point/aliyun_only_mother.pt")
        wandb.finish()
    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(
                checkpoint.optimizer_scheduler_state_dict
            )
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

    def test(self):
        all_pred_ans=[]
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, 1):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)

            #train_loss, train_interaction = self.train_epoch()
            test_lossn,test_f1_scoren,test_acc_scoren,test_precision_scoren,test_recall_scoren = self.eval_epoch(self.test_data)

            print("|  {:>4} |    {:.5f} |     {:.5f}    |     {:.5f}    |   {:.5f} |   {:.5f} |   {:.5f} |   {:.5f} |   {:.5f} |    {:.5f}    |    {:.5f}    |"
                  .format(epoch, 0.0, 0.0, 0.0 ,0.0, 0.0, test_lossn, test_f1_scoren,  test_acc_scoren,test_precision_scoren,test_recall_scoren))


        for callback in self.callbacks:
            callback.on_train_end()      