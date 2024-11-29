from ..data_casual import input_list_test, input_list_train, output_list_test, output_list_train
import torch
ckpt_dir = "./"
tokenizer_path = "./tokenizer.model"
temperature = 0.6
top_p = 0.9
max_seq_len = 1024
max_batch_size = 4
max_gen_len = 1023


def extracting_steering_vector(generator, layer=28, iter=2000):
    generator.change_activation_layer(layer)
    generator.change_activation_bool(True)
    max_gen_len= generator.model.params.max_seq_len - 1
    neg_activation_vector_dic = {}
    pos_activation_vector_dic = {}
    loss = 0
    for i in range(iter):
        value, activation_vec_list = generator.chat_completion([input_list_train[i]],
                                           max_gen_len=max_gen_len,
                                           top_p=top_p,
                                           temperature=temperature)
        value_numerical = "1" if value[0]["generation"]["content"] == "T" else 0
        if value_numerical == output_list_train[i]:
            pos_activation_vector_dic[i] = activation_vec_list[0] 
            loss += 1/iter
        else :
            neg_activation_vector_dic[i] = activation_vec_list[0]
            loss += 1/len(input_list_train)
    pos_vec_sum = torch.zeros([len(neg_activation_vector_dic[1])])
    for ind, item in pos_activation_vector_dic.items():
        pos_vec_sum += 1/len(pos_activation_vector_dic)*item


    neg_vec_sum = torch.zeros([len(neg_activation_vector_dic[1])])
    for ind, item in neg_activation_vector_dic.items():
        neg_vec_sum += 1/len(neg_activation_vector_dic)*item

    steering_vector = pos_vec_sum-neg_vec_sum
    return steering_vector, loss

def calc_loss_steering_vector(generator,layer=28, iter=len(input_list_test), multiplier=1):
    assert iter <= len(input_list_test), f"Test set is smaller than {iter}"
    steering_vec = torch.load(f"./steering_vectors/vector{layer}")
    steering_vec = multiplier*steering_vec
    generator.change_activation_vector(steering_vec)
    generator.change_activation_layer(layer)
    generator.change_activation_bool(False)
    loss = 0
    for sample in range(iter):
        value, activations = generator.chat_completion([input_list_test[sample]],
                                          max_gen_len=max_gen_len,
                                           top_p=top_p,
                                           temperature=temperature)
        assert value[0]["generation"]["content"] =="T" or value[0]["generation"]["content"]=="F", f"Steering vector{layer} is too large" 
        value2 = 1 if value[0]["generation"]["content"]=="T" else 0
        loss += 1/iter*abs(int(output_list_test[sample])-int(value2))

    return loss

