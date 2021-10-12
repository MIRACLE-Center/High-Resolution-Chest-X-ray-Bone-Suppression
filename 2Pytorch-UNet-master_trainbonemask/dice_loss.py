import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    ##继承了Function说明，他是一个层神经元
    def forward(self, input, target):
        ##他一次只输入一个数，不是整个input，看下面的定义，其实全是target
        ##input是用网络算完之后的生成的，在这个模型中是input而已，原来的网络中，他是output
        #target 就是grandtruth
        self.save_for_backward(input, target)
        #给backward用的参数
        self.inter = torch.dot(input.view(-1), target.view(-1)) + 0.0001
        # 这个dot是内积，得到的是一个数字,view,就是reshape.-1时表示自动生成1维
        self.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = 2 * self.inter.float() / self.union.float()
        ##这个是dice系数的公式，2倍的相同的像素个数/他们各自有像素值的点的总个数
        ##因此这个应该是0 1 分布的mask
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        ##可以自己选择让这个 层有没有grad
        input, target = self.saved_variables
        #上面的forward 已经给它存好了，来使用
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
        ##看是不是用cpu跑，如果是就生成cuda格式
    for i, c in enumerate(zip(input, target)):
        ##列成降一个维的了（3d->2d or 2d->1d），不一定是几维。
        # zip是让他们两个一一对应
        #因为我的输入可能是一组文件，所以他可能是一个
        s = s + DiceCoeff().forward(c[0], c[1])
        ##因为最后生成的是两
        return s / (i + 1)
        ##取平均值
