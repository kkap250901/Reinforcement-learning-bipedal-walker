# TQC step 1 : Prediction of N distributions 
# TQC step 2 : Pooling 
# TQC step 3 : Truncation of smallest kN atoms which gives the truncated mixture 
# TQC step 4 : Get the target distribution which is given through the discounting and the shifting 

"""
Implementation from https://github.com/SamsungLabs/tqc_pytorch/tree/master/tqc with changes to the initial noise
Removal of eval environments and fixing of bug where critic was being optimised before the actor, this was a problem
as the actor chanegs the critic weights

Based on the original paper https://arxiv.org/pdf/2005.04269.pdf#page=9&zoom=100,0,0

"""

#########################---------------------Imports-------------------------------------------##############################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch 
import gym 
import numpy as np
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid
from gym import spaces
import copy
import argparse
from pyvirtualdisplay import Display

# This is for rescaling the noise
LOG_STD_MIN_MAX = (-20, 2)


# Loss funciton
def quantile_huber_loss_f(quantiles, samples):
	pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
	abs_pairwise_delta = torch.abs(pairwise_delta)
	huber_loss = torch.where(abs_pairwise_delta > 1,
							 abs_pairwise_delta - 0.5,
							 pairwise_delta ** 2 * 0.5)

	n_quantiles = quantiles.shape[2]
	tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
	loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
	return loss

# Rescaling action to -1, 1 for the continous environment
class RescaleAction(gym.ActionWrapper):
	def __init__(self, env, a, b):
		assert isinstance(env.action_space, spaces.Box), (
			"expected Box action space, got {}".format(type(env.action_space)))
		assert np.less_equal(a, b).all(), (a, b)
		super(RescaleAction, self).__init__(env)
		self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
		self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
		self.action_space = spaces.Box(low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype)

	def action(self, action):
		assert np.all(np.greater_equal(action, self.a)), (action, self.a)
		assert np.all(np.less_equal(action, self.b)), (action, self.b)
		low = self.env.action_space.low
		high = self.env.action_space.high
		action = low + (high - low)*((action - self.a)/(self.b - self.a))
		action = np.clip(action, low, high)
		return action


# The Multi layer perceptron (NN) for the critic 
class Mlp(Module):
	def __init__(self,input_size,hidden_sizes,output_size):
		super().__init__()
		# TODO: initialization
		self.fcs = []
		in_size = input_size
		for i, next_size in enumerate(hidden_sizes):
			fc = Linear(in_size, next_size)
			self.add_module(f'fc{i}', fc)
			self.fcs.append(fc)
			in_size = next_size
		self.last_fc = Linear(in_size, output_size)

	def forward(self, input):
		h = input
		for fc in self.fcs:
			h = relu(fc(h))
		output = self.last_fc(h)
		return output

# Possible to improve with ERE or PER (But a different PER as PER performs well on discrete) 
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.transition_names = ('state', 'action', 'next_state', 'reward', 'not_done')
		sizes = (state_dim, action_dim, state_dim, 1, 1)
		for name, size in zip(self.transition_names, sizes):
			setattr(self, name, np.empty((max_size, size)))

	def add(self, state, action, next_state, reward, done):
		values = (state, action, next_state, reward, 1. - done)
		for name, value in zip(self.transition_names, values):
			getattr(self, name)[self.ptr] = value

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		names = self.transition_names
		return (torch.FloatTensor(getattr(self, name)[ind]).to(DEVICE) for name in names)


# Critic
class Critic(Module):
	def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
		super().__init__()
		self.nets = []
		self.n_quantiles = n_quantiles
		self.n_nets = n_nets
		for i in range(n_nets):
			net = Mlp(state_dim + action_dim, [512, 512, 512], n_quantiles)
			self.add_module(f'qf{i}', net)
			self.nets.append(net)

	def forward(self, state, action):
		sa = torch.cat((state, action), dim=1)
		quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
		return quantiles

# Actor
class Actor(Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.action_dim = action_dim
		self.net = Mlp(state_dim, [256, 256], 2 * action_dim)

	def forward(self, obs):
		mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
		log_std = log_std.clamp(*LOG_STD_MIN_MAX)

		if self.training:
			std = torch.exp(log_std)
			tanh_normal = TanhNormal(mean, std)
			action, pre_tanh = tanh_normal.rsample()
			log_prob = tanh_normal.log_prob(pre_tanh)
			log_prob = log_prob.sum(dim=1, keepdim=True)
		else:  # deterministic eval without log_prob computation
			action = torch.tanh(mean)
			log_prob = None
		return action, log_prob

	def select_action(self, obs):
		obs = torch.FloatTensor(obs).to(DEVICE)[None, :]
		action, _ = self.forward(obs)
		action = action[0].cpu().detach().numpy()
		return action

# Tanh to scale actions and good choice as actions lie in 01,1
class TanhNormal(Distribution):
	def __init__(self, normal_mean, normal_std):
		super().__init__()
		self.normal_mean = normal_mean
		self.normal_std = normal_std
		self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=DEVICE),
									  torch.ones_like(self.normal_std, device=DEVICE))
		self.normal = Normal(normal_mean, normal_std)

	def log_prob(self, pre_tanh):
		log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
		result = self.normal.log_prob(pre_tanh) - log_det
		return result

	def rsample(self):
		pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
		return torch.tanh(pretanh), pretanh

# Trainer class
class Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		top_quantiles_to_drop,
		target_entropy,
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target

################=--------Changed this from torch.zeros --> torch.ones so that the walker is explorign more intially----###
		self.log_alpha = torch.ones((1,), requires_grad=True, device=DEVICE)


		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.target_entropy = target_entropy

		self.quantiles_total = critic.n_quantiles * critic.n_nets

		self.total_it = 0

	def train(self, replay_buffer, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]
# 			print(self.quantiles_total-self.top_quantiles_to_drop)
			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target)

		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()
	
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# --- Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self.total_it += 1


## Episode length means the max number of timesteps per 
max_timesteps_per_ep = 2000

# How many episodes you want the video 
video_every = 25
display = Display(visible=0,size=(600,600))
display.start()

def main(args):

	# Making environment Pls chaneg this to easy or harcore 
	env = gym.make('BipedalWalker-v3')
	env.seed(42)
	env = RescaleAction(env, -1., 1.)

	# For the purposes of recording video, also ep_id - 1 as we start with ep 1 
	env = gym.wrappers.Monitor(env, "./video", video_callable=lambda ep_id: ep_id % video_every == 0, force=True)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	replay_buffer = ReplayBuffer(state_dim, action_dim)
	actor = Actor(state_dim, action_dim).to(DEVICE)
	critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
	critic_target = copy.deepcopy(critic)

	top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets

	trainer = Trainer(actor=actor,
					  critic=critic,
					  critic_target=critic_target,
					  top_quantiles_to_drop=top_quantiles_to_drop,
					  discount=args.discount,
					  tau=args.tau,
					  target_entropy=-np.prod(env.action_space.shape).item())

	reward_list = []
	state, done = env.reset(), False
	episode_return = 0
	episode_timesteps = 0
	episode_num = 1
	log_f = open("agent-log","w+")
	actor.train()

	# Max timestep
	for t in range(int(100000000)):
		action = actor.select_action(state)
		next_state, reward, done, _ = env.step(action)
		episode_timesteps += 1

#################--------------------------------------Reward scaling tried---------------------------------#########################################
		# From td3-fork 
		# reward_scaled = reward.copy()
		# ###### Reward Scaling
		# if reward_scaled < -100.:
		# 	reward_scaled = -5.
		# else:
		# 	reward_scaled = reward_scaled * 5
#############################################################################################################################

		replay_buffer.add(state, action, next_state, reward, done)

		state = next_state
		episode_return += reward

		# Train agent after collecting sufficient data
		if t >= args.batch_size:
			trainer.train(replay_buffer, args.batch_size)

		if done or episode_timesteps >= max_timesteps_per_ep:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
			# Reset environment
			reward_list.append(episode_return)
			log_f.write('episode: {}, reward: {}\n'.format(episode_num, episode_return))
			log_f.flush()
			episode_num += 1
			state, done = env.reset(), False
			episode_return = 0
			episode_timesteps = 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--n_quantiles", default=25, type=int)
	parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
	parser.add_argument("--n_nets", default=5, type=int)
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
	parser.add_argument("--log_dir", default='.')
	parser.add_argument("--prefix", default='')
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	args = parser.parse_args()

	main(args)