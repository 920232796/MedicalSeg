

from medpy import metric
from tqdm import tqdm 
import torch 
import setproctitle
from medical_seg.evaluation import average_metric

class Trainer:
    def __init__(self, network, train_loader, val_loader, optimizer, loss_func, metric, sliding_window_infer,
                    val_internal=1, epochs=1000, k_fold=0, is_show_image=False, device="cpu",
                    model_save_dir="./save/", lr_scheduler=None, start_val=0, log_all_metric=False, task_name="task:0") -> None:
        self.network = network
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.epochs = epochs
        self.k_fold = k_fold
        self.is_show_image = is_show_image
        self.loss_func = loss_func
        self.val_internal = val_internal
        self.val_loader = val_loader
        self.device = device 
        self.metric = metric
        self.sliding_window_infer = sliding_window_infer
        self.model_save_dir = model_save_dir
        self.lr_scheduler = lr_scheduler
        self.scaler = torch.cuda.amp.GradScaler()
        self.start_val = start_val
        self.log_all_metric = log_all_metric
        self.task_name = task_name

    def train(self):

        class_name = self.metric.class_name
        self.best_metric = {c + "_Dice": 0.0 for c in class_name}

        for epoch in range(self.epochs):
            
            print(f"epoch is {epoch}, k_fold is {self.k_fold}")
            setproctitle.setproctitle("{}_{}_{}".format(self.task_name, self.k_fold, epoch))
            self.network.train()
            epoch_loss_1 = 0.0
            for image, label in tqdm(self.train_loader, total=len(self.train_loader)):
                self.optimizer.zero_grad()
                modality_num = image.shape[1]

                image = image.to(self.device)
                label = label.to(self.device).long()
                # print(image.shape, label.shape)
                if self.is_show_image:
                    import matplotlib.pyplot as plt 
                    for i in range(len(label[0])):
                        print(i)
                        for j in range(modality_num):

                            plt.subplot(1, modality_num + 1, j+1)
                            plt.imshow(image[0, j, i], cmap="gray")
                        
                        plt.subplot(1, modality_num+1, modality_num+1)
                        plt.imshow(label[0, i], cmap="gray")
                        plt.show()
                    
                    continue

                with torch.cuda.amp.autocast():
                    pred_1 = self.network(image)
                    loss = self.loss_func(pred_1, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # pred_1 = self.network(image)
                # loss = self.loss_func(pred_1, label)

                # loss.backward()
                # self.optimizer.step()

                epoch_loss_1 += loss.item()

            if self.lr_scheduler is not None :
                self.lr_scheduler.step()
            print(f"epoch_loss_1 is {epoch_loss_1}")

            if (epoch + 1) % self.val_internal == 0 and (epoch + 1) > (self.start_val):
                metric = self.test_and_save(epoch)

        return metric, self.best_metric 


    def metric_dice_sum(self, metric):
        metric_sum = 0.0
        for c in self.metric.class_name:
            metric_sum += metric[c + "_Dice"]
        
        return metric_sum

    def test_and_save(self, epoch):

        metric = self.test_model()    
        best_metric_sum = self.metric_dice_sum(self.best_metric)
        metric_sum = self.metric_dice_sum(metric)

        if best_metric_sum < metric_sum:
            self.best_metric = metric
            save_path = f"{self.model_save_dir}model_{self.k_fold}_best.bin"
            torch.save(self.network.state_dict(), save_path)
            print(f"best model is saved .{save_path}")

        print(f"metric is {metric} \n best metric is {self.best_metric}")

        with open(f"{self.model_save_dir}epoch_res.txt", "a+") as f:
            f.write(f"epoch:{epoch+1}, metric:{metric}")
            f.write("\n")

        if epoch == self.epochs - 1:
            # 保存最终模型
            save_path = f"{self.model_save_dir}model_{self.k_fold}_epoch_{epoch + 1}.bin"
            torch.save(self.network.state_dict(), save_path)

        return metric

    def test_model(self):
    # 训练完毕进行测试。
        self.network.eval()
        end_metric = []
        for image, label in tqdm(self.val_loader, total=len(self.val_loader)):
            image = image.to(self.device)
            label = label.to(self.device).long()
            with torch.no_grad():

                pred_1 = self.sliding_window_infer(image, self.network)
                pred_1 = pred_1.argmax(dim=1)
                
                end_metric.append(self.metric.run(pred=pred_1.cpu().numpy(), label=label.cpu().numpy()))
        
        if self.log_all_metric:
            with open(f"{self.model_save_dir}all_metric_epoch.txt", "a+") as f:
                for m in end_metric:
                    f.write(f"metric:{m}")
                    f.write("\n")
                print("~~~~~~~~~~~~~~~~~~~")

        end_metric = average_metric(end_metric)
    
        return end_metric