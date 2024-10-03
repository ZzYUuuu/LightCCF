import random

import scipy.sparse as sp
import numpy as np
class Data(object):
    def __init__(self, args):
        self.args = args
        self.path = self.args.dataset_path + args.dataset
        self.filetype = self.args.dataset_type
        self.num_users = 0
        self.num_items = 0
        self.num_nodes = 0
        self.load_data_and_create_sp()
        if int(args.sparsity_test) == 1:
            self.split_test_dict, self.split_state = self.create_sparsity_split()

    def load_data_and_create_sp(self):
        train_path = self.path + "/train" + self.filetype
        test_path = self.path + "/test" + self.filetype

        self.unique_train_users, self.train_users, self.train_items, self.train_pos_len, self.train_num_inter, self.train_dict = self.read_file(train_path)
        self.unique_test_users,  self.test_users,  self.test_items,  self.test_pos_len,  self.test_num_inter, self.test_dict = self.read_file(test_path)
        assert len(self.train_users) == len(self.train_items)

        self.num_users += 1
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items

        # U*I
        self.train_mat = sp.coo_matrix((np.ones(len(self.train_users)), (self.train_users, self.train_items)), shape=[self.num_users, self.num_items])
        self.test_mat  = sp.coo_matrix((np.ones(len(self.test_users)),  (self.test_users, self.test_items)),   shape=[self.num_users, self.num_items])

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))

    def read_file(self, file_name):
        inter_users, inter_items, unique_user, user_dict = [], [], [], {}
        pos_length = []
        num_inter = 0
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]

                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))

                unique_user.append(user_id)

                inter_users.extend([user_id] * len(pos_id))
                inter_items.extend(pos_id)

                pos_length.append(len(pos_id)) # [10, 20, 10, 15]
                num_inter += len(pos_id)

                for i in range(0, len(pos_id)):
                    if i == 0:
                        user_dict[user_id] = [pos_id[i]]
                    else:
                        user_dict[user_id].append(pos_id[i])

                line = f.readline()

        return np.array(unique_user), np.array(inter_users), np.array(inter_items), pos_length, num_inter, user_dict

    def random_create_user_pos_neg(self):
        pairs = []

        for i in range(len(self.train_users)):
            user = self.train_users[i]
            pos_items = self.train_dict[user]
            if len(pos_items) == 0:
                continue

            pos_item = self.train_items[i]
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in pos_items:
                    break
            pairs.append([user, pos_item, neg_item])
        return np.array(pairs)
    def random_create_user_pos_neg_cl(self):
        pairs = []
        for i in range(len(self.train_users)):
            user = self.train_users[i]
            pos_items = self.train_dict[user]
            if len(pos_items) == 0:
                continue

            pos_item = self.train_items[i]
            while True:
                pos_item2 = random.sample(pos_items, k=1)[0]
                if pos_item2 != pos_item:
                    break
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in pos_items:
                    break
            pairs.append([user, pos_item, neg_item, pos_item2])
        return np.array(pairs)

    def sparse_adjacency_matrix(self):
        try:
            normal_adjacency = sp.load_npz(self.path + '/pre_Adj.npz')
            print('\t Adjacency matrix exist. Now loading!')
        except:
            print('\t Adjacency matrix not exist. Now constructing!')
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.train_mat.todok()
            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T

            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            normal_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_Adj', normal_adjacency)
            print('\t Adjacency matrix constructed.')
        return normal_adjacency
    def sparse_adjacency_matrix_self(self):
        try:
            normal_adjacency = sp.load_npz(self.path + '/pre_Adj_self.npz')
            print('\t Adjacency matrix exist. Now loading!')
        except:
            print('\t Adjacency matrix not exist. Now constructing!')
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.train_mat.todok()
            # adjacency_matrix[row1:row2, column1:column2]
            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T

            # add self
            adjacency_matrix = adjacency_matrix.todok()
            adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

            # A_hat = D^(-1/2) A D(-1/2)
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            normal_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_Adj_self', normal_adjacency)
            print('\t Adjacency matrix constructed.')
        return normal_adjacency
    def user_item_num(self):
        return self.num_users, self.num_items

    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = dict()

        for uid in all_users:
            train_iids = self.all_positive[uid]
            test_iids = self.test_dict[uid]

            num_iids = len(train_iids) + len(test_iids)

            if num_iids not in user_n_iid.keys():
                user_n_iid[num_iids] = [uid]
            else:
                user_n_iid[num_iids].append(uid)

        split_uids = list()
        temp = []
        count = 1
        fold = 3
#         fold = 4
        n_count = self.train_num_inter + self.test_num_inter
        n_rates = 0
        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.334 * (self.train_num_inter + self.test_num_inter):
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
    def get_user_pos_items(self, users):
        self.train_mat_csr = self.train_mat.tocsr()
        positive_items = []
        for user in users:
            positive_items.append(self.train_mat_csr[user].nonzero()[1])
        return positive_items
