import torch.nn as nn
from taxonomy import SECONDARY_TASKS
from transformers import AutoModel

class BertMultiTask(nn.Module):
    def __init__(
        self, model_name, extra_layer_sizes=[], dropout_rate=0.1, finetune: bool = False
    ):
        super(BertMultiTask, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.layers = nn.ModuleList()
        if not finetune:
            self.name = f"{model_name.split('/')[-1]}_all_tasks_{'_'.join(map(str, extra_layer_sizes))}"
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            self.name = f"{model_name.split('/')[-1]}_finetune_all_tasks_{'_'.join(map(str, extra_layer_sizes))}"
            for param in self.bert.parameters():
                param.requires_grad = True

        prev_size = self.bert.config.hidden_size
        for size in extra_layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        self.reg_head = nn.Linear(prev_size, 1)  # for education value

        self.classification_heads = nn.ModuleDict()
        for task_name, id_map in SECONDARY_TASKS.items():
            self.classification_heads[task_name] = nn.Linear(prev_size, len(id_map))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        x = pooled_output
        for layer in self.layers:
            x = layer(x)

        reg_output = self.reg_head(x).squeeze(-1)
        classes_outputs = {}
        for task, head in self.classification_heads.items():
            classes_outputs[task] = head(x)

        return reg_output, classes_outputs

    def model_unique_name(self) -> str:
        return self.name
