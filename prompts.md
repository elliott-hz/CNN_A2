please read CNN_A2/README.md.

read all the markdown files in the folder of CNN_A2/requirements/.

now please read and understand the whole process of all 6 classification experiments in the package of CNN_A2/experiments/classification_ResNet50_baseline.py
classification_ResNet50_reduced_v1.py
classification_ResNet50_reduced_v2.py
classification_ResNet50_deeper_v1.py
classification_ResNet50_deeper_v2.py
classification_ResNet50_deeper_v3.py









please read CNN_A2/README.md.

read all the markdown files in the folder of CNN_A2/requirements/.

now please read and understand the whole process of all 3 YOLOv8 experiments in the package of CNN_A2/experiments/ (start with 'detection_YOLOv8' python files)








please read CNN_A2/README.md.

read all the markdown files in the folder of CNN_A2/requirements/.

now please read and understand the whole process of all 3 FasterRCNN experiments in the package of CNN_A2/experiments/ (start with 'detection_FasterRCNN' python files)





please read CNN_A2/README.md. read all the markdown files in the folder of CNN_A2/requirements/. now please read and understand the whole process of all 3 FasterRCNN experiments in the package of CNN_A2/notebooks/detection_FasterRCNN/detection_FasterRCNN_v1.ipynb and FasterRCNN_modules.ipynb

let me know if you finish reading and understanding




我现在的需求是：认真看下面的2个notebook代码 CNN_A2/notebooks/detection_FasterRCNN/detection_FasterRCNN_v1.ipynb and FasterRCNN_modules.ipynb

尤其是Cell5 训练的部分代码，，我希望 这个代码支持 记录checkpoint，断了，重新训练的时候可以从断了那一轮重新训练（参数也使用训练断之前的那一轮训练结束的参数继续）

我再详细的说： 
1、假设 detection_FasterRCNN_v1.ipynb 执行从cell1->cell4都完成了，cell5执行到了N轮，，但是服务器断了或者停了，，，（这个时候服务器应该已经保存的很多checkpoints，应该保存了N-1轮的模型参数和数据（training_history等等） 
2、我重新启动服务器的时候，我又从cell1-cell4再执行一次（这个也是必要的东西），然后cell5从N轮重新训练 traiing_history.csv 追加记录。结束cell5也能正常进行 validation等等 
3、支持多次断，多次重新执行。


另外：1. Checkpoint保存策略：每10个epoch保存一次； 2. 旧Checkpoint清理：否: 保留所有历史checkpoint用于回滚 3: Resume时的配置验证：允许配置变化，仅警告（更灵活） 4: CSV文件处理：直接追加到现有CSV（简单）


直接修改吧