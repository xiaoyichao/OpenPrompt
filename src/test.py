import os
import sys
os.chdir(sys.path[0])
sys.path.append("../")

from openprompt4distilbert.data_utils import InputExample
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "4",
    "3",
    "2",
    "1",   
]
dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "客厅",
        text_b = "客厅的电视背景墙我用的乳胶漆",
        label = 1,
    ),
    InputExample(
        guid = 1,
        text_a = "卫生间",
        text_b = "厨房的小白砖是我的最爱",
        label = 0,
    ),
]

from openprompt4distilbert.plms import load_plm
distilbert_torch_path = "/data/search_opt_model/topk_opt/distilbert/distilbert_torch"
plm, tokenizer, model_config, WrapperClass = load_plm("distilbert", distilbert_torch_path)

# plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

from openprompt4distilbert.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = '用户的搜索词是{"placeholder":"text_a"}，展示的内容是{"placeholder":"text_b"}，用户认为搜索词和内容的相关性是{"mask"}分。',
    tokenizer = tokenizer,
)

from openprompt4distilbert.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "4": ["收藏"],
        "3": ["评论或者点赞"],
        "2": ["有效点击"],
        "1": ["无点击或无效点击"],
    },
    tokenizer = tokenizer,
)

from openprompt4distilbert import PromptForClassification
promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)

from openprompt4distilbert import PromptDataLoader
data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

import torch
from transformers4token import  AdamW, get_linear_schedule_with_warmup
use_cuda = True
# use_cuda = False
from transformers4token import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

print("names: ", [n for n, p in promptModel.plm.named_parameters()])
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in promptModel.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in promptModel.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

print("names: ", [n for n, p in promptModel.template.named_parameters()])
# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    # {'params': [p for n,p in promptModel.template.named_parameters() if "raw_embedding" not in n]}
    {'params': [p for n,p in promptModel.template.named_parameters()]}
]

optimizer1 = AdamW(optimizer_grouped_parameters1, lr=0)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-1/1024)

# making zero-shot inference using pretrained MLM with prompt
for epoch in range(3):
    # ## train
    promptModel.train()


    tot_loss = 0
    for step, inputs in enumerate(data_loader):
        if use_cuda:
            # device = "cuda:0"
            # promptModel.to(device)
            inputs = inputs.cuda()
            promptModel = promptModel.cuda()

        logits = promptModel(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        # print(promptModel.template.soft_embeds.grad)
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        print(f"epoch {epoch} - step {step}: ", "loss:", loss.item(), "tot_loss:", tot_loss/(step+1))

promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        if use_cuda:
            batch = batch.cuda()
            promptModel=  promptModel.cuda()
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        print(classes[preds])
# predictions would be 1, 0 for classes 'positive', 'negative'