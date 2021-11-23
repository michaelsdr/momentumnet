import numpy as np
import matplotlib.pyplot as plt
import torch
from momentumnet import transform_to_momentumnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seq_size = 15
d = 512
n_layers = 12
m_batch_sizes = [128, 256, 512, 1024]
batch_sizes = [128, 256, 512]  # Without momentum, increased batch size would saturate the memory of the gpu


def train(net, src, tgt):
    loss = (net(src, tgt) ** 2).mean()
    loss.backward()


def profile(net, batch_sizes):
    res = []
    for bs in batch_sizes:
        src = torch.rand((seq_size, bs, d)).to(device)
        tgt = torch.rand((seq_size, bs, d)).to(device)

        train(net, src, tgt)
        used_mem = torch.cuda.max_memory_allocated(device=device)
        res.append(used_mem * 9.5367431640625e-7)
    res = np.asarray(res) / 1000
    return res


transformer = torch.nn.Transformer(num_encoder_layers=n_layers, num_decoder_layers=n_layers).to(device)
mtransformer = transform_to_momentumnet(transformer, sub_layers=["encoder.layers", "decoder.layers"], gamma=0.95,
                                        use_backprop=False, keep_first_layer=False)

res_mtransformer = profile(mtransformer, m_batch_sizes)
res_transformer = profile(transformer, batch_sizes)

fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

ax.plot(res_transformer,  linewidth=5, color='darkblue', label='Transformer')
ax.plot(res_mtransformer,  linewidth=5, color='red', label='Momentum Transformer')
ax.set_xticks(np.arange(len(m_batch_sizes)))
ax.set_xticklabels(m_batch_sizes)
ax.set_ylabel('Memory (Gib)')
ax.set_xlabel('Batch size')
ax.set_yticks([4, 8, 16])
ax.set_title('Fixed sequence length of 15')
fig.tight_layout(pad=0.6)
plt.legend(loc='best', ncol=2,  handlelength=1, handletextpad=.35, borderpad=0.15)
plt.show()
