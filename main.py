import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
from solver import Solver


def classify(args, leave_one_num=-1):
    torch.cuda.empty_cache()
    model_name = args.method_name + '_' + args.dataset + \
        '_sensor_num' + str(args.sensor_num) + \
        '_subject_idx' + str(leave_one_num)
    solver = Solver(args, leave_one_num=leave_one_num, model_name=model_name)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    acc_test_vec = np.zeros(args.max_epoch)
    if not args.eval_only:
        for t in tqdm(range(args.max_epoch)):
            solver.train()
            acc_test_vec[t] = solver.test()
            solver.save_model()
            print('Test target acc:', 100.0 * acc_test_vec[t], '%')
    return solver.test_ensemble(), acc_test_vec, model_name


def save_kd_dataset(args, leave_one_num=-1):
    torch.cuda.empty_cache()
    model_name = args.method_name + '_' + args.dataset + \
        '_sensor_num' + str(args.sensor_num) + \
        '_subject_idx' + str(leave_one_num)
    solver = Solver(args, leave_one_num=leave_one_num, model_name=model_name)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    return solver.save_kd_dataset()


def convert_args_to_bool(args):
    args.eval_only = (args.eval_only in ['True', True])
    return args


def process(args):
    print(args)
    model_name = f'{args.method_name}_{args.dataset}_sensor_num_{args.sensor_num}_temperature_{args.temperature}'
    sub_num_dict = {'ENABL3S': 10, 'DSADS': 8}
    sub_num = sub_num_dict[args.dataset]
    acc_s = np.zeros(sub_num)
    acc_t = np.zeros(sub_num)
    acc_val_mat = np.zeros((sub_num, args.max_epoch))
    for i in range(sub_num):
        print('Test ', i)
        (acc_s[i], acc_t[i]), acc_val_mat[i], _ = classify(
            args, leave_one_num=i)
    if not args.eval_only:
        result_dir = 'results'
        os.makedirs(result_dir, exist_ok=True)
        np.savetxt(f"{result_dir}/final_acc_{model_name}.csv",
                   np.transpose(np.c_[acc_s, acc_t]), delimiter=",")
        np.savetxt(f"{result_dir}/acc_mat_{model_name}.csv",
                   np.transpose(acc_val_mat), delimiter=",")
    print('{}: Mean of test acc in the source domain:'.format(
        model_name), np.mean(acc_s))
    print('{}: Mean of test acc in the target domain:'.format(
        model_name), np.mean(acc_t))


def train(args):
    '''Train the teacher network'''
    process(args)
    ''' Save knowledge distillation datasets '''
    sub_num_dict = {'ENABL3S': 10, 'DSADS': 8}
    sub_num = sub_num_dict[args.dataset]
    for i in range(sub_num):
        save_kd_dataset(args, leave_one_num=i)
    ''' Knowledge distillation: Change the method name to method_name + KD'''
    if not ('KD' in args.method_name):
        args.method_name = f'{args.method_name}KD'
        process(args)


def test(args):
    process(args)


if __name__ == '__main__':
    print(os.getcwd())
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DSADS', metavar='N',
                        help='Dataset is ENABL3S or DSADS?')
    parser.add_argument('--sensor_num', type=int, default=0, metavar='N',
                        help='Different combination of sensors: \
                        0: all sensors')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoint', metavar='N',
                        help='checkpoint directory')
    parser.add_argument('--method_name', default='EDHKD',
                        help='check the method name')
    parser.add_argument('--eval_only', default=True,
                        help='evaluation only option')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                        help='how many epochs')
    parser.add_argument('--dis_c_ratio', type=float, default=5.0, metavar='LR',
                        help='ratio of classifier discrepancy (default: 5.0)')
    parser.add_argument('--dis_f_ratio', type=float, default=5.0, metavar='LR',
                        help='ratio of feature discrepancy (default: 1e-2)')
    parser.add_argument('--ent_ratio', type=float, default=1e-2, metavar='LR',
                        help='ratio of mcd loss (default: 1e-1)')
    parser.add_argument('--num_k', type=int, default=4, metavar='N',
                        help='hyper paremeter for generator update')
    parser.add_argument('--optimizer', type=str, default='adam',
                        metavar='N', help='which optimizer')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--temperature', type=int, default=1, metavar='N',
                        help='temperature paremeter for the softmax distribution of the knowledge distillation')

    args = parser.parse_args()
    args = convert_args_to_bool(args)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
    for dataset_name in ['ENABL3S', 'DSADS']:
        args.dataset = dataset_name
        if args.eval_only:
            test(args)
        else:
            args.method_name = 'EDH'
            train(args)
