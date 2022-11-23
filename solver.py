import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import Generator, Classifier, DomainClassifier
from datasets.dataset_read import dataset_read, save_pseudo_dataset, load_target_data, read_pseudo_dataset

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


def Solver(args, leave_one_num=-1, model_name=''):
    method_dict = {'EDH': EDH,
                   'EDHKD': EDHKD}
    return method_dict[args.method_name](args, leave_one_num, model_name)


class EDH(object):
    def __init__(self, args, leave_one_num=-1, model_name='', num_G=5, num_C=25, num_D=0):
        self.args = args
        self.batch_size = args.batch_size
        self.num_k = args.num_k
        self.checkpoint_dir = args.checkpoint_dir
        self.lr = args.lr

        self.num_G = num_G  # number of generators
        self.num_C = num_C  # number of classifiers
        self.num_D = num_D  # number of domain classifiers
        self.leave_one_num = leave_one_num
        self.model_name = model_name

        self.data_train, self.data_val, self.data_test = dataset_read(
            self.batch_size, is_resize=True, leave_one_num=self.leave_one_num,
            dataset=args.dataset, sensor_num=args.sensor_num)

        self.net_dict = self.init_model()
        if args.eval_only:
            self.load_model()
        else:
            self.set_optimizer(which_opt=args.optimizer, lr=args.lr)

    def init_model(self):
        G_list = []
        for i in range(self.num_G):
            G_list.append(Generator(dataset=self.args.dataset,
                                    sensor_num=self.args.sensor_num).to(device))
        C_list = []
        for i in range(self.num_C):
            C_list.append(Classifier(dataset=self.args.dataset).to(device))
        D_list = []
        for i in range(self.num_D):
            D_list.append(DomainClassifier().to(device))

        return {'G': G_list, 'C': C_list, 'D': D_list}

    def save_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                torch.save(self.net_dict[key][i], '{}/{}_{}_{}.pt'.format(
                    self.args.checkpoint_dir, self.model_name, key, i))

    def load_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                if 'cpu' in device_name:
                    self.net_dict[key][i] = torch.load('{}/{}_{}_{}.pt'.format(
                        self.args.checkpoint_dir, self.model_name, key, i),
                        map_location=lambda storage, loc: storage)
                else:
                    file_name = '{}/{}_{}_{}.pt'.format(
                        self.args.checkpoint_dir, self.model_name, key, i)
                    self.net_dict[key][i] = torch.load(file_name)
        print('Loaded file: {}'.format(file_name))

    def train_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                self.net_dict[key][i].train()

    def eval_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                self.net_dict[key][i].eval()

    def step_model(self, keys):
        for key in keys:
            for i in range(len(self.opt_dict[key])):
                self.opt_dict[key][i].step()
        self.reset_grad()

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        self.opt_dict = {}
        for key in self.net_dict.keys():
            self.opt_dict.update({key: []})
            for i in range(len(self.net_dict[key])):
                if which_opt == 'momentum':
                    opt = optim.SGD(self.net_dict[key][i].parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
                elif which_opt == 'adam':
                    opt = optim.Adam(self.net_dict[key][i].parameters(),
                                     lr=lr, weight_decay=0.0005)
                else:
                    raise Exception("Unrecognized optimization method.")
                self.opt_dict[key].append(opt)

    def reset_grad(self):
        for key in self.opt_dict.keys():
            for i in range(len(self.opt_dict[key])):
                self.opt_dict[key][i].zero_grad()

    def calc_classifier_discrepancy_loss(self, output_list):
        loss = 0.0
        num_C_for_G = int(self.num_C/self.num_G)
        for r in range(self.num_G):
            mean_output_t = torch.mean(
                torch.stack(output_list[num_C_for_G * r:num_C_for_G * (r + 1)]), dim=0)
            for c in range(num_C_for_G):
                loss += self.discrepancy(output_list[num_C_for_G * r + c],
                                         mean_output_t)
        return loss

    def calc_output_list(self, img):
        # return parallel_calc_output_list(self.net_dict['G'], self.net_dict['C'], img)
        feat_list = [None for _ in range(self.num_G)]
        output_list = [None for _ in range(self.num_C)]
        num_C_for_G = int(self.num_C / self.num_G)
        for r in range(len(self.net_dict['G'])):
            feat_list[r] = self.net_dict['G'][r](img)
        for r in range(len(self.net_dict['G'])):
            for c in range(num_C_for_G):
                output_list[r * num_C_for_G + c] = self.net_dict['C'][r *
                                                                      num_C_for_G + c](feat_list[r])
        return feat_list, output_list

    def calc_mean_output(self, img):
        _, output_list = self.calc_output_list(img)
        return torch.mean(torch.stack(output_list, dim=0), dim=0)

    def train_DA(self, img_s, img_t, label_s):
        # 1: Minimize the classification error of source data
        feat_s_list, output_s_list = self.calc_output_list(img_s)
        loss = self.calc_source_loss(output_s_list, label_s) \
            - self.args.dis_f_ratio * \
            self.calc_feature_discrepency_loss(feat_s_list)
        loss.backward()
        self.step_model(['G', 'C'])

        # 2: Maximize the discrepancy of classifiers
        _, output_s_list = self.calc_output_list(img_s)
        _, output_t_list = self.calc_output_list(img_t)
        loss = self.calc_source_loss(output_s_list, label_s) \
            - self.args.dis_c_ratio * self.calc_classifier_discrepancy_loss(output_t_list) \
            + self.args.ent_ratio * self.calc_output_ent(output_t_list)

        loss.backward()
        self.step_model(['C'])

        # 3: Minimize the discrepancy of classifiers by training feature extractor
        for i in range(self.num_k):
            feat_t_list, output_t_list = self.calc_output_list(img_t)
            loss = self.calc_classifier_discrepancy_loss(output_t_list) \
                + self.args.ent_ratio * self.calc_output_ent(output_t_list) \
                - self.calc_feature_discrepency_loss(feat_t_list)\

            loss.backward()
            self.step_model(['G'])

    def train(self):
        self.train_model()
        torch.cuda.manual_seed(1)
        for batch_idx, data in enumerate(self.data_train):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']

            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            img_s = Variable(img_s.to(device))
            img_t = Variable(img_t.to(device))
            label_s = Variable(label_s.long().to(device)).squeeze()

            self.reset_grad()
            self.train_DA(img_s, img_t, label_s)

    def test_ensemble(self):
        self.load_model()
        acc_s = self.test(set_name='S')
        print('Final test source acc:', 100.0 * acc_s, '%')
        acc_t = self.test(set_name='T')
        print('Final test target acc:', 100.0 * acc_t, '%')
        return acc_s, acc_t

    def test(self, set_name='T'):
        self.eval_model()
        correct_val, size_val = self.calc_correct_and_size(
            self.data_val, set_name=set_name)
        correct_test, size_test = self.calc_correct_and_size(
            self.data_test, set_name=set_name)
        correct = correct_val + correct_test
        size = size_val + size_test
        return float(correct) / float(size)

    def calc_correct_and_size(self, data_eval, set_name='T'):
        correct = 0.0
        size = 0.0
        for batch_idx, data in enumerate(data_eval):
            img = data[set_name]
            label = data[set_name + '_label']
            img, label = Variable(img.to(device)), Variable(
                label.long().to(device))
            feat_list, output_list = self.calc_output_list(img=img)

            output_vec = torch.stack(output_list)
            pred_ensemble = output_vec.data.max(dim=-1)[1]
            pred_ensemble = torch.mode(pred_ensemble, dim=0)[0]
            k = label.data.size()[0]
            correct += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        return correct, size

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))

    def calc_feature_discrepency_loss(self, feat_list):
        criterion_consistency = nn.L1Loss().to(device)
        mean_feat = torch.mean(torch.stack(feat_list), dim=0)
        loss = criterion_consistency(feat_list[0], mean_feat)
        for feat_s in feat_list[1:]:
            loss += criterion_consistency(feat_s, mean_feat)
        return loss

    def calc_source_loss(self, output_s_list, label_s):
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(output_s_list[0], label_s)
        for output_s in output_s_list[1:]:
            loss += criterion(output_s, label_s)
        return loss

    def ent(self, out_t):
        out_t = F.softmax(out_t, dim=-1)
        loss_ent = - \
            torch.mean(torch.sum(out_t * (torch.log(out_t + 1e-5)), dim=-1))
        return loss_ent

    def calc_output_ent(self, output_t_list):
        loss = 0.0
        for output_t in output_t_list:
            loss += self.ent(output_t)
        return loss

    def save_kd_dataset(self):
        self.load_model()
        data_target = load_target_data(leave_one_num=self.leave_one_num,
                                       dataset=self.args.dataset,
                                       sensor_num=self.args.sensor_num)
        x_name_list = ['x_t_train']
        y_name_list = ['y_pseudo_t_train']
        for i in range(len(x_name_list)):
            # The code below replaced the hard target label y_t_train by the soft target pseudo label
            data_target[y_name_list[i]] = self.estimate_pseudo_label(
                data_target[x_name_list[i]])
        save_pseudo_dataset(data_target, leave_one_num=self.leave_one_num,
                            dataset=self.args.dataset,
                            sensor_num=self.args.sensor_num)

    def estimate_pseudo_label(self, X_np):
        self.eval_model()
        X = Variable(torch.from_numpy(X_np).float().to(device))
        _, output_list = self.calc_output_list(img=X)
        # (net_num, batch_size, mode_num)
        output_vec = torch.stack(output_list)
        # (batch_size, mode_num)
        pred_ensemble = output_vec.data.mean(dim=0)
        pred_ensemble = pred_ensemble.cpu().detach().numpy()
        print(pred_ensemble.shape)
        return pred_ensemble


class EDHKD(EDH):
    def __init__(self, args, leave_one_num=-1, model_name='', num_G=1, num_C=1, num_D=0):
        EDH.__init__(self, args=args, leave_one_num=leave_one_num, model_name=model_name,
                     num_G=num_G, num_C=num_C, num_D=num_D)
        self.data_train, self.data_test = read_pseudo_dataset(
            self.batch_size, is_resize=True, leave_one_num=self.leave_one_num,
            dataset=args.dataset, sensor_num=args.sensor_num)
        self.temperature = args.temperature
        print(f'Temperature value: {self.temperature}')
        print('Model EDHKD')

    def train(self):
        '''
        TODO: Add knowledge distillation
        '''
        self.train_model()
        torch.cuda.manual_seed(1)
        for batch_idx, (img_t, pseudo_label_t) in enumerate(self.data_train):
            img_t = Variable(img_t.to(device))
            pseudo_label_t = Variable(
                pseudo_label_t.long().to(device)).squeeze()
            self.reset_grad()
            self.train_DA(img_t, pseudo_label_t)

    def train_DA(self, img_t, pseudo_label_t):
        # 1: Minimize the classification error of source data
        _, output_t_list = self.calc_output_list(img_t)
        loss = soft_cross_entropy(
            output_t_list[0]/self.temperature, pseudo_label_t/self.temperature)
        loss.backward()
        self.step_model(['G', 'C'])

    def test_ensemble(self):
        self.load_model()
        acc_t = self.test()
        print('Final test target acc:', 100.0 * acc_t, '%')
        # For knowledge distillation, there is no source data,
        # and thus acc_s is invalid and set to 0 by default
        acc_s = 0
        return acc_s, acc_t

    def test(self):
        self.eval_model()
        correct, size = self.calc_correct_and_size(self.data_test)
        return float(correct) / float(size)

    def calc_correct_and_size(self, data_loader):
        correct = 0.0
        size = 0.0
        for _, (img, label) in enumerate(data_loader):
            img, label = Variable(img.to(device)), Variable(
                label.long().to(device))
            _, output_list = self.calc_output_list(img=img)
            output_vec = torch.stack(output_list)
            pred_ensemble = output_vec.data.max(dim=-1)[1]
            pred_ensemble = torch.mode(pred_ensemble, dim=0)[0]
            k = label.data.size()[0]
            correct += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        return correct, size


def soft_cross_entropy(predicted, target):
    target = F.softmax(target, dim=1)
    predicted = F.log_softmax(predicted, dim=1)
    return -(target * predicted).sum(dim=1).mean()
