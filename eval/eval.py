import torch
ckpt_dir = "./"
tokenizer_path = "./tokenizer.model"
temperature = 0.6
top_p = 0.9
max_seq_len = 1024
max_batch_size = 4
max_gen_len = 1023


def extracting_steering_vector(generator, data, layer=28, iter=2000):
    input_list_train = data[0]
    output_list_train = data[1]
    instruct_vec = torch.load(f"./steering_vectors/instruct_vector{layer}.pt")
    generator.change_activation_layer(layer)
    generator.change_activation_bool(True)
    generator.change_activation_vector(torch.tensor(4096*[0]))
    max_gen_len= generator.model.params.max_seq_len - 1
    neg_activation_vector_dic = {}
    pos_activation_vector_dic = {}
    loss = 0
    for i in range(iter):
        value, activation_vec_list = generator.chat_completion([input_list_train[i]],
                                           max_gen_len=max_gen_len,
                                           top_p=top_p,
                                           temperature=temperature)
        value_numerical = "1" if value[0]["generation"]["content"] == "T" else "0"
        if value_numerical == output_list_train[i]:
            pos_activation_vector_dic[i] = activation_vec_list[0] - instruct_vec 
        else :
            neg_activation_vector_dic[i] = activation_vec_list[0] - instruct_vec
            loss += 1/len(input_list_train)
    pos_vec_sum = torch.tensor(4096*[0.0])
    for ind, item in pos_activation_vector_dic.items():
        pos_vec_sum += 1/len(pos_activation_vector_dic)*item


    neg_vec_sum = torch.tensor(4096*[0.0])
    for ind, item in neg_activation_vector_dic.items():
        neg_vec_sum += 1/len(neg_activation_vector_dic)*item

    steering_vector = pos_vec_sum-neg_vec_sum
    return steering_vector, loss

def calc_loss_steering_vector(generator, steering_vec, data, layer=28, iter=1000, multiplier=1):
    input_list_test = data[0]
    output_list_test = data[1]
    assert iter <= len(input_list_test), f"Test set is smaller than {iter}"
    steering_vec = multiplier*steering_vec
    generator.change_activation_vector(steering_vec)
    generator.change_activation_layer(layer)
    generator.change_activation_bool(False)
    loss = 0
    wrong_class = 0 
    for sample in range(iter):
        value, activations = generator.chat_completion([input_list_test[sample]],
                                          max_gen_len=max_gen_len,
                                           top_p=top_p,
                                           temperature=temperature)
        if value[0]["generation"]["content"]=="T":
            value2 = 1
            loss += 1/iter*abs(int(output_list_test[sample])-int(value2))
        elif value[0]["generation"]["content"]=="F":
            value2 = 0
            loss += 1/iter*abs(int(output_list_test[sample])-int(value2))
        else:
            loss += 1/iter
            wrong_class += 1
    print(f"{wrong_class} questions were wrongly classified")
    return loss

