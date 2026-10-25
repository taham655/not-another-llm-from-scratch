import math
import torch



class Adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None 
        if closure is not None: 
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None: 
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                grad = p.grad

                # 1. weight decay: shrink p toward 0 (uses plain lr, not alpha_t)
                p.mul_(1 - lr * weight_decay)

                # 2. direction memory: m = beta1*m + (1-beta1)*g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3. size memory: v = beta2*v + (1-beta2)*g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4. bias-corrected step size, then p -= alpha_t * m / (sqrt(v) + eps)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)


        
        return loss


