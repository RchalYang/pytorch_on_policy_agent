import collections
import copy
import torch
from torch.distributions import Categorical
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np

from utils.torch_utils import device,Tensor, Tensor_zeros_like
import utils.math_utils as math_utils
from .a2c_agent import A2CAgent

class TRPOAgent(A2CAgent):
    def __init__(self,args,env_wrapper, continuous):
        """
        Instantiate a TRPO agent
        """
        super(TRPOAgent, self).__init__(args,env_wrapper, continuous)
                                        
        self.max_kl = args.max_kl
        self.cg_damping = args.cg_damping
        self.cg_iters = args.cg_iters
        self.residual_tol = args.residual_tol

        self.algo="trpo"

    def mean_kl_divergence(self, model):
        """
        Returns an estimate of the average KL divergence between a given model and self.policy_model
        """
        actprob = model(self.observations).detach() + 1e-8
        old_actprob = self.policy(self.observations)

        return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()

    def hessian_vector_product(self, vector):
        """
        Returns the product of the Hessian of the KL divergence and the given vector
        """
        self.policy.zero_grad()
        mean_kl_div = self.mean_kl_divergence(self.policy)
        
        kl_grad_vector = torch.autograd.grad(mean_kl_div, self.policy.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad_vector])
        grad_vector_product = torch.sum(kl_grad_vector * vector)
        second_order_grad = torch.autograd.grad(grad_vector_product, self.policy.parameters())
        
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in second_order_grad])

        return fisher_vector_product + self.cg_damping * vector

    def conjugate_gradient(self, b):
        """
        Returns F^(-1) b where F is the Hessian of the KL divergence
        """
        p = b.clone()
        r = b.clone()
        x = Tensor_zeros_like(p)
        rdotr = r.double().dot(r.double())
            
        for _ in range(self.cg_iters):
            z = self.hessian_vector_product(p).squeeze(0)
            v = (rdotr / p.double().dot(z.double())).float()

            x += v * p
            r -= v * z

            newrdotr = r.double().dot(r.double())
            mu = newrdotr / rdotr
            
            p = r + mu.float() * p
            rdotr = newrdotr
            if rdotr < self.residual_tol:
                break
        return x

    def surrogate_loss(self, theta):
        """
        Returns the surrogate loss w.r.t. the given parameter vector theta
        """
        theta = theta.detach()
        new_model = copy.deepcopy(self.policy)
        vector_to_parameters(theta, new_model.parameters())

        prob_new = new_model(self.observations).gather(1, self.actions).detach()
        prob_old = self.policy(self.observations).gather(1, self.actions).detach() + 1e-8

        return -torch.mean((prob_new / prob_old) * self.advantage )

    def linesearch(self, x, fullstep, expected_improve_rate):
        """
        Returns the parameter vector given by a linesearch
        """
        accept_ratio = .1
        max_backtracks = 10
        fval = self.surrogate_loss(x)
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            print("Search number {}...".format(_n_backtracks + 1))
            stepfrac = float(stepfrac)
            xnew     = x + stepfrac * fullstep
            newfval  = self.surrogate_loss(xnew)
            actual_improve = fval - newfval

            expected_improve = expected_improve_rate * stepfrac
            
            ratio = actual_improve / expected_improve
            
            if ratio > accept_ratio and actual_improve > 0:
                return xnew
        return x.detach()

    def _optimize(self, observations, actions, discounted_rewards):
        """
        TRPO Update
        """
        
        self.observations, self.actions, self.discounted_rewards = observations, actions, discounted_rewards

        # Generate Tensor
        self.observations       = Tensor(self.observations)
        self.actions            = Tensor(self.actions).long().unsqueeze(1)
        self.discounted_rewards = Tensor(self.discounted_rewards).unsqueeze(1)

        # Calculate Advantage & Normalize it
        baseline = self.value(self.observations).detach()  
        self.advantage = self.discounted_rewards - baseline
        self.advantage = (self.advantage - self.advantage.mean()) / (self.advantage.std() + 1e-8)

        # Surrogate loss with Entropy
        action_dists = self.policy(self.observations)
        new_p = action_dists.gather(1, self.actions)
        old_p = new_p.detach() + 1e-8
        prob_ratio = new_p / old_p

        entropy = -torch.sum(action_dists*action_dists.log(), 1).mean()
        
        surrogate_loss = - torch.mean(prob_ratio * self.advantage) - self.entropy_para * entropy

        # Calculate the gradient of the surrogate loss
        self.policy.zero_grad()
        surrogate_loss.backward()
        policy_gradient = parameters_to_vector([p.grad for p in self.policy.parameters()]).squeeze(0).detach()
        
        # ensure gradient is not zero
        if policy_gradient.nonzero().size()[0]:
            # Use Conjugate gradient to calculate step direction
            step_direction = self.conjugate_gradient(-policy_gradient)
            
            # line search for step 
            shs = .5 * step_direction.dot(self.hessian_vector_product(step_direction))
            
            lm = torch.sqrt(shs / self.max_kl)
            fullstep = step_direction / lm

            gdotstepdir = -policy_gradient.dot(step_direction)
            theta = self.linesearch(parameters_to_vector(self.policy.parameters()).detach(), fullstep, gdotstepdir / lm)
            # Update parameters of policy model
            old_model = copy.deepcopy(self.policy)
            old_model.load_state_dict(self.policy.state_dict())

            if any(np.isnan(theta.cpu().detach().numpy())):
                print("NaN detected. Skipping update...")
            else:
                vector_to_parameters(theta, self.policy.parameters())

            kl_old_new = self.mean_kl_divergence(old_model)
            print( 'KL:{:10} , Entropy:{:10}'.format(kl_old_new.item(), entropy.item()))

        else:
            print("Policy gradient is 0. Skipping update...")

        self.value.zero_grad()
        values = self.value(self.observations)
        criterion = torch.nn.MSELoss()
        critic_loss = criterion(values, self.discounted_rewards )
        critic_loss.backward()
        self.value_optimizer.step()
        print("MSELoss for Value Net:{}".format(critic_loss.item()))
        
