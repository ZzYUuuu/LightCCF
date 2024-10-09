import numpy as np
import torch
import utility.tools as tools

def testing(model, args, dataset, device):
    model = model.eval()

    topK = eval(args.top_K)

    model_results ={'recall': np.zeros(len(topK)), 'ndcg': np.zeros(len(topK))}

    with torch.no_grad():
        test_users = list(dataset.test_dict.keys())
        user_list, true_list, rating_list = [], [], []
        num_batch = len(test_users) // int(args.test_batch_size) + 1

        for batch_users in tools.mini_batch(test_users, batch_size=int(args.test_batch_size)):
            exclude_users, exclude_items = [], []

            test_batch_pos = [dataset.test_dict[u] for u in batch_users]

            for i, u in enumerate(batch_users):
                if u in dataset.train_dict:
                    exclude_users.extend([i] * len(dataset.train_dict[u]))
                    exclude_items.extend(dataset.train_dict[u])
            batch_users_device = torch.Tensor(batch_users).long().to(device)

            rating = model.get_rating_for_test(batch_users_device)

            rating[exclude_users, exclude_items] = -1

            _, rating_k = torch.topk(rating, k=max(topK))

            rating = rating.cpu()
            del rating

            user_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            true_list.append(test_batch_pos)
        # batch finish, calculate recall and ndcg
        assert num_batch == len(user_list)

        enum_list = zip(rating_list, true_list)
        results = []
        for single_list in enum_list:
            results.append(test_single_batch(single_list, topK))

        i = 0
        for result in results:
            i += 1
            model_results['recall'] += result['recall']
            model_results['ndcg'] += result['ndcg']
            print('\t Steps %d/%d: recall = %.3f, ndcg = %.3f' % (i, num_batch, float(result['recall'][1]), float(result['ndcg'][1])), end='\r')
        model_results['recall'] /= float(len(test_users))
        model_results['ndcg'] /= float(len(test_users))

        return model_results

def testing_group(model, args, dataset, device, my_dicts):
    model = model.eval()

    topK = eval(args.top_K)

    model_results ={'recall': np.zeros(len(topK)), 'ndcg': np.zeros(len(topK))}

    with torch.no_grad():
        test_users = list(my_dicts.keys())
        user_list, true_list, rating_list = [], [], []
        num_batch = len(test_users) // int(args.test_batch_size) + 1

        for batch_users in tools.mini_batch(test_users, batch_size=int(args.test_batch_size)):
            exclude_users, exclude_items = [], []

            # pair(train_batch_user, train_batch_pos)
            test_batch_pos = [my_dicts[u] for u in batch_users]

            for i, u in enumerate(batch_users):
                exclude_users.extend([i] * dataset.train_pos_len[u])
                exclude_items.extend(dataset.train_dict[u])
            batch_users_device = torch.Tensor(batch_users).long().to(device)

            # model func to calculate the score
            rating = model.get_rating_for_test(batch_users_device)
            # print(batch_users)

            rating[exclude_users, exclude_items] = -1

            # topk return values, indices
            _, rating_k = torch.topk(rating, k=max(topK))

            rating = rating.cpu()
            del rating

            user_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            true_list.append(test_batch_pos)
        # batch finish, calculate recall and ndcg
        assert num_batch == len(user_list)

        enum_list = zip(rating_list, true_list)
        results = []
        for single_list in enum_list:
            results.append(test_single_batch(single_list, topK))

        # i = 0
        for result in results:
            i += 1
            model_results['recall'] += result['recall']
            model_results['ndcg'] += result['ndcg']
        model_results['recall'] /= float(len(test_users))
        model_results['ndcg'] /= float(len(test_users))

        return model_results

def test_single_batch(single_list, topK):
    pred_items = single_list[0].numpy()
    true_items = single_list[1]

    pred_item_label = pred_to_label(pred_items, true_items)

    recall, ndcg = [], []
    for k_size in topK:
        recall.append(recall_k(pred_item_label, k_size, true_items))
        ndcg.append(ndcg_k(pred_item_label, k_size, true_items))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}


def pred_to_label(pred_items, true_items):
    pred_item_label = []
    for i in range(len(true_items)):
        true_item = true_items[i]
        pred_item = pred_items[i]

        pred = list(map(lambda x: x in true_item, pred_item)) 

        pred = np.array(pred).astype("float")

        pred_item_label.append(pred) # [True, False, True, ...]

    return np.array(pred_item_label).astype("float")

def recall_k(pred, k_size, true):
    pred_k = pred[:, : k_size].sum(1)
    recall_num = np.array([len(true[i]) for i in range(len(true))])
    recall = np.sum(pred_k / recall_num)
    return recall

def ndcg_k(pred, k_size, true):
    assert len(pred) == len(true)

    pred_matrix = pred[:, :k_size]
    true_matrix = np.zeros((len(pred_matrix), k_size))
    for i, items in enumerate(true):
        length = k_size if k_size <= len(items) else len(items)
        true_matrix[i, :length] = 1

    dcg  = np.sum(pred_matrix * (1. / np.log2(np.arange(2, k_size + 2))), axis=1)
    idcg = np.sum(true_matrix * (1. / np.log2(np.arange(2, k_size + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def sparsity_test(dataset, args, model, device):
    sparsity_results = []
    model = model.eval()
    # top-20, 40, ..., 100
    topK = eval(args.top_K)

    with torch.no_grad():
        for users in dataset.split_test_dict:
            model_results = {
                # 'precision': np.zeros(len(topK)),
                'recall': np.zeros(len(topK)),
                # 'hit': np.zeros(len(topK)),
                'ndcg': np.zeros(len(topK))
            }
            users_list, rating_list, ground_true_list = [], [], []
            num_batch = len(users) // int(args.test_batch_size) + 1

            for batch_users in tools.mini_batch(users, batch_size=int(args.test_batch_size)):
                exclude_users, exclude_items = [], []
                all_positive = dataset.get_user_pos_items(batch_users)
                ground_true = [dataset.test_dict[u] for u in batch_users]

                batch_users_device = torch.Tensor(batch_users).long().to(device)

                rating = model.get_rating_for_test(batch_users_device)

                # Positive items are excluded from the recommended list
                for i, items in enumerate(all_positive):
                    exclude_users.extend([i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_users, exclude_items] = -1

                # get the top-K recommended list for all users
                _, rating_k = torch.topk(rating, k=max(topK))

                rating = rating.cpu()
                del rating

                users_list.append(batch_users)
                rating_list.append(rating_k.cpu())
                ground_true_list.append(ground_true)

            assert num_batch == len(users_list)
            enum_list = zip(rating_list, ground_true_list)

            results = []
            for single_list in enum_list:
                results.append(test_single_batch(single_list, topK))

            for result in results:
                model_results['recall'] += result['recall']
                # model_results['precision'] += result['precision']
                model_results['ndcg'] += result['ndcg']

            model_results['recall'] /= float(len(users))
            # model_results['precision'] /= float(len(users))
            model_results['ndcg'] /= float(len(users))
            sparsity_results.append(model_results)

    return sparsity_results
