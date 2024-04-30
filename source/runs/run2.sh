cd ../main/

export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9992 --dataset="PACS" --subdataset="2" > ../../ckps/PACS/2/2.out &
sleep 30
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9992 --dataset="PACS" --subdataset="0" &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9992 --dataset="PACS" --subdataset="1" &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9992 --dataset="PACS" --subdataset="3" &
sleep 3