import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pytorch_lightning as pl

from transformer import SimpleTransformer
from Process import read_data, create_fields, create_dataset
from Batch import create_masks


parser = argparse.ArgumentParser()
parser.add_argument('-src_data', required=True)
parser.add_argument('-trg_data', required=True)
parser.add_argument('-src_lang', required=True)
parser.add_argument('-trg_lang', required=True)
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-SGDR', action='store_true')
parser.add_argument('-epochs', type=int, default=2)
parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-heads', type=int, default=8)
parser.add_argument('-dropout', type=int, default=0.1)
parser.add_argument('-batchsize', type=int, default=1500)
parser.add_argument('-printevery', type=int, default=100)
parser.add_argument('-lr', type=int, default=0.0001)
parser.add_argument('-load_weights')
parser.add_argument('-create_valset', action='store_true')
parser.add_argument('-max_strlen', type=int, default=80)
parser.add_argument('-floyd', action='store_true')
parser.add_argument('-checkpoint', type=int, default=0)
parser.add_argument('-device', type=str, default='cpu')
opt = parser.parse_args()

dims = 512
heads = 8
N = 6


read_data(opt)
SRC, TRG = create_fields(opt)
opt.train = create_dataset(opt, SRC, TRG)
src_vocab = len(SRC.vocab)
trg_vocab = len(TRG.vocab)
model = SimpleTransformer(src_vocab, trg_vocab, dims, N, heads)

opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)


class LightningTransformer(pl.LightningModule):
    def __init__(self, src_vocab, target_vocab, dims, N, heads, opt):
        super(LightningTransformer, self).__init__()
        self.transformer = SimpleTransformer(src_vocab, target_vocab, dims, N, heads)
        self.cptime = time.time()
        self.opt = opt
        self.total_loss = 0

    def forward(self, src, target, src_mask, target_mask):
        output = self.transformer(src, target, src_mask, target_mask)
        return output  # don't need to do softmax because loss function will apply softmax

    def training_step(self, batch, batch_idx):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, self.opt)
        preds = model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        self.opt.optimizer.zero_grad()
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=self.opt.trg_pad)
        self.opt.optimizer.step()
        if self.opt.SGDR == True:
            self.opt.sched.step()

        self.total_loss += loss.item()

        if (batch_idx + 1) % self.opt.printevery == 0:
            p = int(100 * (batch_idx + 1) / self.opt.train_len)
            avg_loss = self.total_loss / self.opt.printevery
            # if opt.floyd is False:
            #     print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
            #           ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
            #            "".join(' ' * (20 - (p // 5))), p, avg_loss), end='\r')
            # else:
            #     print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
            #           ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
            #            "".join(' ' * (20 - (p // 5))), p, avg_loss))
            self.total_loss = 0

        if opt.checkpoint > 0 and ((time.time() - self.cptime) // 60) // opt.checkpoint >= 1:
            torch.save(model.state_dict(), 'weights/model_weights')
            self.cptime = time.time()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = LightningTransformer(src_vocab, trg_vocab, dims, N, heads, opt)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


def train_model(model, opt):
    print("training model...")
    model.train()
    start = time.time()

    trainer = pl.Trainer(max_epochs=opt.epochs)
    trainer.fit(model, opt.train)
    # if opt.checkpoint > 0:
    #     cptime = time.time()

    # for epoch in range(opt.epochs):

    # total_loss = 0
    # if opt.floyd is False:
    #     print("   %dm: epoch %d [%s]  %d%%  loss = %s" % \
    #           ((time.time() - start) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')
    #
    # if opt.checkpoint > 0:
    #     torch.save(model.state_dict(), 'weights/model_weights')

    # for i, batch in enumerate(opt.train):



        # print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
        #       ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
        #        avg_loss, epoch + 1, avg_loss))

train_model(model, opt)
# trainer = pl.Trainer()
# trainer.fit(model)
