import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            target = word2number_dict[
                word[word_index + n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch  # (batch num, batch size, n_step) (batch num, batch size)


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict


class TextLSTM(nn.Module):
    def __init__(self, emb_size, n_hidden):
        super(TextLSTM, self).__init__()
        self.hidden_size = n_hidden
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        # self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        # i_t
        self.U_i = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.V_i = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_i = nn.Parameter(torch.Tensor(n_hidden))
        # f_t
        self.U_f = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.V_f = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_f = nn.Parameter(torch.Tensor(n_hidden))
        # c_t
        self.U_c = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.V_c = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_c = nn.Parameter(torch.Tensor(n_hidden))
        # o_t
        self.U_o = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.V_o = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_o = nn.Parameter(torch.Tensor(n_hidden))

        # i_t1
        self.U_i1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.V_i1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_i1 = nn.Parameter(torch.Tensor(n_hidden))
        # f_t1
        self.U_f1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.V_f1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_f1 = nn.Parameter(torch.Tensor(n_hidden))
        # c_t1
        self.U_c1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.V_c1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_c1 = nn.Parameter(torch.Tensor(n_hidden))
        # o_t1
        self.U_o1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.V_o1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_o1 = nn.Parameter(torch.Tensor(n_hidden))

        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        results=[]
        batch_size, n_step,_= X.size()
        h_t0, c_t0, h_t1, c_t1 = (
            torch.zeros(batch_size, self.hidden_size).to(X.device),
            torch.zeros(batch_size, self.hidden_size).to(X.device),
            torch.zeros(batch_size, self.hidden_size).to(X.device),
            torch.zeros(batch_size, self.hidden_size).to(X.device),
            )
        for t in range(n_step):
            x_t = X[:, t, :]
            i_t = torch.sigmoid(x_t @ self.U_i + h_t0 @ self.V_i + self.b_i)  # 输入门
            g_t = torch.tanh(x_t @ self.U_c + h_t0 @ self.V_c + self.b_c)  # 输入门
            f_t = torch.sigmoid(x_t @ self.U_f + h_t0 @ self.V_f + self.b_f)  # 遗忘门
            o_t = torch.sigmoid(x_t @ self.U_o + h_t0 @ self.V_o + self.b_o)  # 输出门
            c_t = f_t * c_t0 + i_t * g_t  # 记忆更新
            h_t = o_t * torch.tanh(c_t)  # 输出门
        # hidden_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # cell_state = torch.zeros(1, len(X), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
            x_t1 = h_t  #上层LSTM模型输出作为下层模型输入
            i_t1 = torch.sigmoid(x_t1 @ self.U_i1 + h_t1 @ self.V_i1 + self.b_i1)  # 输入门
            g_t1 = torch.tanh(x_t1 @ self.U_c1 + h_t1 @ self.V_c1 + self.b_c1)  # 输入门
            f_t1 = torch.sigmoid(x_t1 @ self.U_f1 + h_t1 @ self.V_f1 + self.b_f1)  # 遗忘门
            o_t1 = torch.sigmoid(x_t1 @ self.U_o1 + h_t1 @ self.V_o1 + self.b_o1)  # 输出门
            c_t1 = f_t1 * c_t1 + i_t1 * g_t1  # 记忆更新
            h_t1 = o_t1 * torch.tanh(c_t1)  # 输出门
        # X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]
            results.append(h_t.unsqueeze(0))
        # outputs, (_, _) = self.LSTM(X, (hidden_state, cell_state))
        #outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        results= torch.cat(results, dim=0)
        outputs=results[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


def train_LSTMlm(emb_size, n_hidden):
    model = TextLSTM(emb_size, n_hidden)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()
            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)
            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target) * 128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch + 1}.ckpt')


def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target) * 128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell
    batch_size = 128  # batch size
    learn_rate = 0.0005
    all_epoch = 5  # the all epoch for training
    emb_size = 256  # embeding size
    save_checkpoint_epoch = 5  # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt')  # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    # print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    print("\nTrain the LSTMLM……………………")
    train_LSTMlm(emb_size, n_hidden)

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)
