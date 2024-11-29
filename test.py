from typing import List, Optional

import fire

from llama import Dialog, Llama
import torch.distributed as dist
import torch
import os
from data_casual import output_list_train, input_list_train

ckpt_dir = "./"
tokenizer_path = "./tokenizer.model"
temperature = 0.6
top_p = 0.9
max_seq_len = 1024
max_batch_size = 4
max_gen_len= None

generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len= 1024, max_batch_size= 4, activation=True, activation_layer=12)
print(generator)




#value, activations = generator.chat_completion([input_list_train[1142]],
#                                   max_gen_len=max_gen_len,
#                                   top_p=top_p,
#                                   temperature=temperature)
#print(value, activations)

#n = len(input_list_test)

#for i in range(n):
#    value = generator.chat_completion([input_list_test[i]],
#                                      max_gen_len=max_gen_len,
#                                       top_p=top_p,
#                                       temperature=temperature)
#   value2 = 1 if value[0]["generation"]["content"]=="T" else 0
#   loss += 1/n*abs(int(output_list_test[i])-int(value2))

#print(loss)