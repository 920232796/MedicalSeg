
nohup python -u ./spleen_task.py > ./nohup_dir/spleen_task.out 2>&1 &
nohup python -u ./spleen_task_cl.py > ./nohup_dir/spleen_task_cl.out 2>&1 &
nohup python -u ./spleen_task_no_cl.py > ./nohup_dir/spleen_task_no_cl.out 2>&1 &

nohup python -u ./brats2020_task.py > ./nohup_dir/brats2020_task_baseline.out 2>&1 &

nohup python -u ./brats2020_task_CoNet.py > ./nohup_dir/brats2020_task_conet.out 2>&1 &

nohup python -u ./brats2020_task_CoNetMLP.py > ./nohup_dir/brats2020_task_CoNet_MLP.out 2>&1 &

nohup python -u ./brats2020_task_cross_att.py > ./nohup_dir/brats2020_task_cross_att.out 2>&1 &

nohup python -u ./brats2020_task_spatial_trans_net.py > ./nohup_dir/brats2020_task_spatial_trans_net.out 2>&1 &

nohup python -u ./brats_unet_baseline.py > ./nohup_dir/brats_unet_baseline.out 2>&1 &

nohup python -u ./brats2020_task_segresnet.py > ./nohup_dir/brats2020_task_segresnet.out 2>&1 &

nohup python -u ./brats2020_task_transbts.py > ./nohup_dir/brats2020_task_transbts.out 2>&1 &

nohup python -u ./brats2020_task_multi_att_net.py > ./nohup_dir/brats2020_task_multi_att_net.out 2>&1 &

nohup python -u ./brats2020_task_swin_unet.py > ./nohup_dir/brats2020_task_swin_unet.out 2>&1 &




nohup python -u ./meni_task.py > ./nohup_dir/meni_task.out 2>&1 & # 1

nohup python -u ./meni_task_tumor.py > ./nohup_dir/meni_task_tumor.out 2>&1 & # 1

nohup python -u ./meni_task_shuizhong.py > ./nohup_dir/meni_task_shuizhong.out 2>&1 & # 1

nohup python -u ./meni_task_conet.py > ./nohup_dir/meni_task_conet.out 2>&1 &

nohup python -u ./meni_task_cross_att.py > ./nohup_dir/meni_task_cross_att.out 2>&1 &

nohup python -u ./meni_task_cross_att_v2.py > ./nohup_dir/meni_task_cross_att_v2.out 2>&1 &

nohup python -u ./meni_task_spatial_trans_net.py > ./nohup_dir/meni_task_spatial_trans_net.out 2>&1 &

nohup python -u ./meni_spatial_trans_net_cpc.py > ./nohup_dir/meni_spatial_trans_net_cpc.out 2>&1 &

nohup python -u ./meni_task_vnet.py > ./nohup_dir/meni_task_vnet.out 2>&1 &

nohup python -u ./meni_task_segresnet.py > ./nohup_dir/meni_task_segresnet.out 2>&1 &

nohup python -u ./meni_task_transbts.py > ./nohup_dir/meni_task_transbts.out 2>&1 &

nohup python -u ./meni_task_multi_att_net.py > ./nohup_dir/meni_task_multi_att_net.out 2>&1 &

nohup python -u ./meni_task_multi_att_net_v2.py > ./nohup_dir/meni_task_multi_att_net_v2.out 2>&1 &

nohup python -u ./meni_task_multi_att_net_v3_cpc.py > ./nohup_dir/meni_task_multi_att_net_v3_cpc.out 2>&1 &

nohup python -u ./meni_task_multi_att_net_v3_cpc.py > ./nohup_dir/meni_task_multi_att_net_v3_cpc_data_aug.out 2>&1 &

nohup python -u ./meni_task_multi_att_net_v3_cpc.py > ./nohup_dir/meni_task_multi_att_net_v4.out 2>&1 &

nohup python -u ./meni_task_multi_attention.py > ./nohup_dir/meni_task_multi_attention_v2.out 2>&1 &



nohup python -u ./meni_task_swin_unet.py > ./nohup_dir/meni_task_swin_unet.out 2>&1 &


#

nohup python -u ./prostate_task_unet.py > ./nohup_dir/prostate_task_unet.out 2>&1 &

nohup python -u ./prostate_task_segresnet.py > ./nohup_dir/prostate_task_segresnet.out 2>&1 &

nohup python -u ./prostate_task_transbts.py > ./nohup_dir/prostate_task_transbts.out 2>&1 &


nohup python -u ./prostate_task_multi_attention.py > ./nohup_dir/prostate_task_multi_attention.out 2>&1 &

nohup python -u ./prostate_task_unet_patch.py > ./nohup_dir/prostate_task_unet_patch.out 2>&1 &


nohup python -u ./prostate_task_multi_attention_patch.py > ./nohup_dir/prostate_task_multi_attention_patch.out 2>&1 &
