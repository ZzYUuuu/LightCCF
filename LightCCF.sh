%run main.py --dataset douban-book --encoder MF --train_batch_size 2048 --tau 0.28 --ssl_lambda 1.0 --log rs

%run main.py --dataset tmall --encoder MF --train_batch_size 4096 --tau 0.24 --ssl_lambda 5.0 --log rs

%run main.py --dataset amazon-book --encoder LightGCN --gcn_layer 3 --train_batch_size 4096 --tau 0.22 --ssl_lambda 5.0 --log rs
