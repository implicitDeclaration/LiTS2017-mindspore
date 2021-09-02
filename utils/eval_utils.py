from mindspore import nn

top1 = nn.TopKCategoricalAccuracy(1)
top5 = nn.TopKCategoricalAccuracy(5)

function = {
    1: top1,
    5: top5
}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    res = []
    for k in topk:
        function[k].clear()
        function[k].update(output, target)
        top = function[k].eval()*100
        res.append(top)
    return res
