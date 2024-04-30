cd ../main/

export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9991 --dataset="PACS" --subdataset="1" > ../../ckps/PACS/1/1.out &
sleep 30
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9991 --dataset="PACS" --subdataset="0" &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9991 --dataset="PACS" --subdataset="2" &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9991 --dataset="PACS" --subdataset="3" &
sleep 3