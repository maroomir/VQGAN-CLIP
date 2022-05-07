from typing import Any

from torch.autograd import Function


class ReplaceGrad(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x_forward = kwargs['forward'] if 'forward' in kwargs else args[0]
        x_backward = kwargs['backward'] if 'backward' in kwargs else args[1]
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_in = grad_outputs[0]
        return None, grad_in.sum_to_size(ctx.shape)

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass


class ClampWithGrad(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input_ = kwargs['input'] if 'input' in kwargs else args[0]
        min_ = kwargs['min'] if 'min' in kwargs else args[1]
        max_ = kwargs['max'] if 'max' in kwargs else args[2]
        ctx.min = min_
        ctx.max = max_
        ctx.save_for_backward(input_)
        return input_.clamp(min_, max_)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_in = grad_outputs[0]
        input_, _ = ctx.saved_tensors
        return grad_in * (grad_in * (input_ - input_.clamp(ctx.min, ctx.max)) >= 0), None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass
