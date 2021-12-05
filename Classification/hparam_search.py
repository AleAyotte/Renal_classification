"""
    @file:              main.py
    @Author:            Alexandre Ayotte

    @Creation Date:     09/2021
    @Last modification: 09/2021

    @Description:       Contain the main function to train a neural network on one of those two dataset: BrainMets
                        and RCC.

    @Reference:         1) https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
"""

from copy import deepcopy
from comet_ml import Experiment
import numpy as np
from torchsummary import summary
from typing import Final

from arg_parser import argument_parser
from constant import  DatasetName, Experimentation, ModelType, SplitName, Tasks
from data_manager.dataset_builder import build_datasets
from model_creation import create_model
from trainer_creation import create_trainer
from trainer.utils import compute_recall
from utils import print_data_distribution, read_api_key, save_hparam_on_comet

BRAIN_METS_TASKS: Final = {Tasks.ARE, Tasks.LRF}
DEFAULT_SHARED_LR_SCALE = 100  # Default rate between shared_lr and lr if shared_lr == 0
MIN_NUM_EPOCH: Final = 2  # Minimum number of epoch to save the experiment with comet.ml
MIN_NUM_TASKS: Final = 2
PRELU_L2: Final = 0  # L2 regularization should not be used when using PRELU activation as recommended by ref 1)
PROJECT_NAME = "dec-2021-hparam"
RCC_TASKS: Final = {Tasks.GRADE, Tasks.MALIGNANCY, Tasks.SUBTYPE}
SINGLE_TASK_EXPERIMENT: Final = [Experimentation.STL_2D, Experimentation.STL_3D]

NUM_CROSS_VAL: Final = 5
NUM_EXPERIMENT: Final = 100


def sample_hparam(hparam_dict):
    hparam_dict.drop_rate = np.random.uniform(0.1, 0.7)
    hparam_dict.in_channels = np.random.randint(16, 64)
    hparam_dict.lr = 10 ** (np.random.uniform(-2, -5))
    hparam_dict.l2 = 10 ** (np.random.uniform(-2, -8))
    hparam_dict.eps = 10 ** (np.random.uniform(-1, -6))

    b_size = np.random.randint(16, 48)
    if b_size >= 28:
        hparam_dict.b_size = int(b_size / 2)
        hparam_dict.num_cumu_batch = 2
    else:
        hparam_dict.b_size = b_size
        hparam_dict.num_cumu_batch = 1
        
    activation = ['ReLU', 'PReLU', 'LeakyReLU', 'ELU']
    hparam_dict.act = activation[np.random.randint(0, 3)]
    args.split_level = np.random.randint(2, 4)


if __name__ == "__main__":
    args = argument_parser()
    experimentation = Experimentation[args.experiment.upper()]
    dataset_name = DatasetName[args.dataset.upper()]
    classification_tasks_list = []

    if experimentation in SINGLE_TASK_EXPERIMENT:
        classification_tasks_list.append(args.task)
    else:
        for task in list(BRAIN_METS_TASKS | RCC_TASKS):
            if args.__getattribute__(task):
                classification_tasks_list.append(task)

    if dataset_name is DatasetName.RCC:
        assert set(classification_tasks_list) <= RCC_TASKS, "Incorrect task choice"
    else:
        assert set(classification_tasks_list) <= BRAIN_METS_TASKS, "Incorrect task choice"

    if experimentation is Experimentation.STL_2D:
        if args.task in ["subtype", "grade"]:
            clin_features = ["Sex", "size", "renal_vein_invasion", "metastasis",
                             "pt1", "pt2", "pt3", "pn1", "pn2", "pn3"]
        else:
            clin_features = ["Age", "Sex", "size"]
        num_clin_features = len(clin_features)
    else:
        clin_features = None
        num_clin_features = 0

    regression_tasks_list = []

    tasks_list = classification_tasks_list + regression_tasks_list

    if experimentation not in SINGLE_TASK_EXPERIMENT:
        assert len(tasks_list) >= MIN_NUM_TASKS, "You have to select at least two task."

    # --------------------------------------------
    #               CREATE DATASET
    # --------------------------------------------
    trainset, validset, testset = build_datasets(classification_tasks=classification_tasks_list,
                                                 clin_features=clin_features,
                                                 dataset_name=dataset_name,
                                                 num_chan=args.num_chan_data,
                                                 regression_tasks=regression_tasks_list,
                                                 split_seed=args.seed,
                                                 testset_name=args.testset)

    args.depth_config = 3
    for i in range(NUM_EXPERIMENT):
        try:
            train_stats = {task: {"recall 0": [], "recall 1": [], "auc": []} for task in classification_tasks_list}
            valid_stats = {task: {"recall 0": [], "recall 1": [], "auc": []} for task in classification_tasks_list}
            test_stats = {task: {"recall 0": [], "recall 1": [], "auc": []} for task in classification_tasks_list}
            sample_hparam(args)

            for _ in range(NUM_CROSS_VAL):
                # --------------------------------------------
                #                 CREATE MODEL
                # --------------------------------------------
                if experimentation is Experimentation.SOFT_SHARING:
                    model_type = ModelType.SHARED_NET
                elif experimentation is Experimentation.LTB:
                    model_type = ModelType.LTB_NET
                elif experimentation is Experimentation.TAG and args.model == "ltb":
                    model_type = ModelType.LTB_NET
                else:
                    model_type = ModelType.STANDARD

                in_shape = tuple(testset[0]["sample"].size()[1:])
                net, num_classes, conditional_prob = create_model(args,
                                                                  experimentation=experimentation,
                                                                  in_shape=in_shape,
                                                                  num_clin_features=num_clin_features,
                                                                  tasks_list=tasks_list)

                if model_type is not ModelType.LTB_NET:
                    summary(net, tuple(trainset[0]["sample"].size()))

                # --------------------------------------------
                #                SANITY CHECK
                # --------------------------------------------
                print_data_distribution(SplitName.TRAIN,
                                        trainset.labels_bincount(),
                                        classification_tasks_list)
                print_data_distribution(SplitName.VALIDATION,
                                        validset.labels_bincount(),
                                        classification_tasks_list)
                print_data_distribution(args.testset.upper(),
                                        testset.labels_bincount(),
                                        classification_tasks_list)
                print("\n")

                # --------------------------------------------
                #                   TRAINER
                # --------------------------------------------

                trainer = create_trainer(args,
                                         experimentation=experimentation,
                                         conditional_prob=None,
                                         model_type=model_type,
                                         num_classes=num_classes,
                                         tasks_list=tasks_list)

                if model_type is ModelType.SHARED_NET:
                    shared_eta_min = args.eta_min * DEFAULT_SHARED_LR_SCALE if args.pretrained else args.eta_min
                    shared_lr = args.lr * DEFAULT_SHARED_LR_SCALE if args.pretrained else args.lr
                    shared_l2 = args.sharing_l2
                elif model_type is ModelType.LTB_NET:
                    shared_eta_min = args.branch_eta
                    shared_lr = args.branch_lr
                    shared_l2 = args.branch_l2
                else:
                    shared_eta_min = 0
                    shared_lr = 0
                    shared_l2 = 0

                trainer.fit(batch_size=args.b_size,
                            device=args.device,
                            eps=args.eps,
                            eta_min=args.eta_min,
                            grad_clip=args.grad_clip,
                            learning_rate=args.lr,
                            l2=PRELU_L2 if args.activation == "PReLU" else args.l2,
                            model=net,
                            mode=args.mode,
                            num_cumulated_batch=args.num_cumu_batch,
                            num_epoch=args.branch_num_epoch if experimentation is Experimentation.LTB else args.num_epoch,
                            optim=args.optim,
                            retrain=args.retrain,
                            shared_eta_min=shared_eta_min,
                            shared_lr=shared_lr,
                            shared_l2=shared_l2,
                            trainset=trainset,
                            t_0=args.branch_num_epoch if experimentation is Experimentation.LTB else args.num_epoch,
                            validset=validset,
                            warm_up_epoch=args.warm_up)

                if experimentation is Experimentation.LTB:
                    parents_list, ltb_task_list = net.freeze_branching()

                    trainer = create_trainer(args,
                                             experimentation=experimentation,
                                             conditional_prob=conditional_prob if len(conditional_prob) > 0 else None,
                                             model_type=ModelType.STANDARD,
                                             num_classes=num_classes,
                                             tasks_list=tasks_list)
                    trainer.fit(batch_size=args.b_size,
                                device=args.device,
                                eps=args.eps,
                                eta_min=args.eta_min,
                                grad_clip=args.grad_clip,
                                learning_rate=args.lr,
                                l2=PRELU_L2 if args.activation == "PReLU" else args.l2,
                                model=net,
                                num_cumulated_batch=args.num_cumu_batch,
                                num_epoch=args.num_epoch,
                                optim=args.optim,
                                trainset=trainset,
                                t_0=args.num_epoch,
                                validset=validset)

                # Print the score in the terminal
                set_list = [trainset, validset, testset]
                split_list = [SplitName.TRAIN, SplitName.VALIDATION, args.testset.upper()]
                stats_dict = [train_stats, valid_stats, test_stats]

                for split_set, split, stats in zip(set_list, split_list, stats_dict):
                    conf, auc = trainer.score(split_set, save_path="")

                    for task in classification_tasks_list:
                        stats[task]["auc"].append(auc[task])
                        recall = compute_recall(conf[task])
                        stats[task]["recall 0"].append(recall[0])
                        stats[task]["recall 1"].append(recall[1])

            # --------------------------------------------
            #                    SCORE
            # --------------------------------------------
            if args.num_epoch > MIN_NUM_EPOCH:
                experiment = Experiment(api_key=read_api_key(),
                                        project_name=PROJECT_NAME,
                                        workspace="aleayotte",
                                        log_env_details=False,
                                        auto_metric_logging=False,
                                        log_git_metadata=False,
                                        auto_param_logging=False,
                                        log_code=False)
                experiment.set_name(f"hparam_{i}")
                experiment.log_code("arg_parser.py")
                experiment.log_code("hparam_search.py")
                experiment.log_code("constant.py")
                experiment.log_code("model_creation.py")
                experiment.log_code("trainer_creation.py")
                experiment.log_code("utils.py")
                experiment.log_code("model/block.py")
                experiment.log_code("model/hard_shared_resnet.py")
                experiment.log_code("model/module.py")
                experiment.log_code("model/mtan.py")
                experiment.log_code("model/neural_net.py")
                experiment.log_code("trainer/trainer.py")
                experiment.log_code("trainer/multi_task_trainer.py")

            else:
                experiment = None

            # Print the score in the terminal
            split_list = [SplitName.TRAIN, SplitName.VALIDATION, args.testset.upper()]
            stats_dict = [train_stats, valid_stats, test_stats]

            for split, stats in zip(split_list, stats_dict):
                for task in classification_tasks_list:
                    auc = float(np.mean(stats[task]["auc"]))
                    recall_0 = float(np.mean(stats[task]["recall 0"]))
                    recall_1 = float(np.mean(stats[task]["recall 1"]))
                    mean_recall = (recall_0 + recall_1) / 2

                    if experiment is not None:
                        name = f"{split} {task.capitalize()}"
                        experiment.log_metric(name=name + " AUC", value=auc)
                        experiment.log_metric(name=name + " Recall 0", value=recall_0)
                        experiment.log_metric(name=name + " Recall 1", value=recall_1)
                        experiment.log_metric(name=name + " Mean Recall", value=mean_recall)

            if experiment is not None:
                hparam = vars(deepcopy(args))
                save_hparam_on_comet(experiment=experiment, args_dict=hparam)
                experiment.log_parameter("Task", "_".join(tasks_list))

        except Exception as e:
            print(e)

