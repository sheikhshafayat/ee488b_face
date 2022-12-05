python trainEmbedNet.py --model iresnet --trainfunc angleproto --save_path exps/finetune21 --nPerClass 2 --gpu 3 --max_epoch 100 --scheduler steplr --batch_size 264 --lr 0.0001 --lr_decay 0.9 --optimizer adamw --weight_decay 0.3
# we first put the pretrained .model file in the exps/finetune21 folder and rename it to model000001.model then run this script
# the details of loss function and models can be found here: https://github.com/sheikhshafayat/ee488b_face