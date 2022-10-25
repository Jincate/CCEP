import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Discriminator(nn.Module):
	def __init__(self, state_dim, num_skills):
		super(Discriminator, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, num_skills)

	def forward(self, state):
		x = F.relu(self.l1(state))
		x = F.relu(self.l2(x))
		return self.l3(x)


class DoubleActor(nn.Module):
	def __init__(self, state_dim, num_skills, action_dim, max_action):
		super(DoubleActor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.l4 = nn.Linear(state_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state, context):
		# su = torch.cat([state, context], dim=1)
		a1 = F.relu(self.l1(state))
		a1 = F.relu(self.l2(a1))
		a1 = self.max_action * torch.tanh(self.l3(a1))
		
		a2 = F.relu(self.l4(state))
		a2 = F.relu(self.l5(a2))
		a2 = self.max_action * torch.tanh(self.l6(a2))

		a = torch.cat((a1, a2), 0)
		index = torch.nonzero(context)
		index = (index[:,0] + index[:,1] * (context.shape[0] - 1 )).reshape(-1)
		return torch.index_select(a, dim = 0, index = index)

class Actor(nn.Module):
	def __init__(self, state_dim, num_skills, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim + num_skills, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
			
		self.max_action = max_action
		

	def forward(self, state, context):
		su = torch.cat([state, context], dim=1)
		a = F.relu(self.l1(su))
		a = F.relu(self.l2(a))
		a = self.max_action * torch.tanh(self.l3(a))
		
		return a

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class DoubleCritic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(DoubleCritic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		# q1 = (q1 * context).sum(dim=1).reshape(-1, 1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		# q2 = (q2 * context).sum(dim=1).reshape(-1, 1)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		# q1 = (q1 * context).sum(dim=1).reshape(-1, 1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		num_skills=10
	):

		self.actor = Actor(state_dim, num_skills, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = DoubleCritic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		
		# self.op_critic = copy.deepcopy(self.critic)
		# self.op_target = copy.deepcopy(self.critic)
		# self.op_optimizer = torch.optim.Adam(self.op_critic.parameters(), lr=3e-4)

		self.disc = Discriminator(state_dim, num_skills).to(device)
		self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=3e-4)
		self.disc_target = copy.deepcopy(self.disc)
		self.num_skills = num_skills
		
		self.single_critic = Critic(state_dim, action_dim).to(device)
		self.single_target = copy.deepcopy(self.single_critic)
		self.single_optimizer = torch.optim.Adam(self.single_critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.total_it = 0
		self.logger = SummaryWriter('logs')

	def select_action(self, state, context):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		context = torch.FloatTensor(context.reshape(1, -1)).to(device)
		return self.actor(state, context).cpu().data.numpy().flatten()

	def state_prob(self, context, state):
		state = torch.FloatTensor(state).to(device).unsqueeze(0)
		context = torch.FloatTensor(context).to(device).unsqueeze(0)
		score = self.disc_target(state)
		prob = F.softmax(score, dim=1)
		prob = prob * context
		prob, _ = torch.max(prob, dim=1, keepdim=False)
		return prob.detach().cpu().numpy()[0]

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state,  action, initial_context, context, next_state, next_context, reward, pseudo_reward, not_done = replay_buffer.sample(batch_size)
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			next_action = (
				self.actor_target(next_state, next_context) + noise
			).clamp(-self.max_action, self.max_action)
			zero_context = np.zeros((batch_size, self.num_skills))
			zero_context[:, 0] = 1
			zero_context = torch.FloatTensor(zero_context.reshape(batch_size, -1)).to(device)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			
			target_Q = torch.min(target_Q1, target_Q2)
			# target_Q4 = torch.min(target_Q1, target_Q2)
			# target_Q1 = pseudo_reward + not_done * self.discount * target_Q1
			# target_Q2 = pseudo_reward + not_done * self.discount * target_Q2
			# target_Q3, target_Q4 = self.single_target(next_state, next_action)
			target_Q = pseudo_reward + not_done * self.discount * target_Q
			# target_Q2 = pseudo_reward + not_done * self.discount * target_Q2
			# target_QQ1, target_QQ2 = self.critic_target(next_state, next_action, zero_context)
			# target_QQ = torch.min(target_QQ1, target_QQ2)
			# target_QQ = reward + not_done * self.discount * target_QQ
		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		# current_Q2 = self.op_critic(state, action)
		# current_Q3, current_Q4 = self.single_critic(state, action)
		# current_QQ1, current_QQ2 = self.critic(state, action, zero_context)
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		#	F.mse_loss(current_QQ1, target_QQ) + F.mse_loss(current_QQ2, target_QQ)
		error = current_Q1 - current_Q2
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		# self.op_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		# self.op_optimizer.step()
		# critic2_loss = F.mse_loss(current_Q3, target_Q3) + F.mse_loss(current_Q4, target_Q4)
		# self.single_optimizer.zero_grad()
		# critic2_loss.backward()
		# self.single_optimizer.step()


		score_vector = self.disc(state)
		context_index = torch.argmax(initial_context, dim=1)
		disc_loss = F.cross_entropy(score_vector, context_index)

		self.disc_optimizer.zero_grad()
		disc_loss.backward()
		self.disc_optimizer.step()
		self.logger.add_scalar("currentQ1/max", current_Q1.max(), self.total_it)
		self.logger.add_scalar("currentQ1/mean", current_Q1.mean(), self.total_it)
		self.logger.add_scalar("currentQ2/max", current_Q2.max(), self.total_it)
		self.logger.add_scalar("currentQ2/mean", current_Q2.mean(), self.total_it)
		# self.logger.add_scalar("currentQ4/max", current_Q4.max(), self.total_it)
		# self.logger.add_scalar("currentQ4/mean", current_Q4.mean(), self.total_it)
		#self.logger.add_scalar("currentQQ1/max", current_QQ1.max(), self.total_it)
		#self.logger.add_scalar("currentQQ1/mean", current_QQ1.mean(), self.total_it)
		self.logger.add_scalar("reward/max", reward.max(), self.total_it)
		self.logger.add_scalar("reward/mean", reward.mean(), self.total_it)
		self.logger.add_scalar("critic_loss", critic_loss, self.total_it)
		self.logger.add_scalar("error/mean", error.mean(), self.total_it)
		self.logger.add_scalar("error/max", error.max(), self.total_it)
		self.logger.add_scalar("error/min", error.min(), self.total_it)
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			context1 = np.zeros((batch_size, self.num_skills))
			context1[:, 0] = 1
			context1 = torch.FloatTensor(context1.reshape(batch_size, -1)).to(device)
			context2 = np.zeros((batch_size, self.num_skills))
			context2[:, 1] = 1
			context2 = torch.FloatTensor(context2.reshape(batch_size, -1)).to(device)
			# context3 = np.zeros((batch_size, self.num_skills))
			# context3[:, 2] = 1
			# context3 = torch.FloatTensor(context3.reshape(batch_size, -1)).to(device)
			# context4 = np.zeros((batch_size, self.num_skills))
			# context4[:, 3] = 1
			# context4 = torch.FloatTensor(context4.reshape(batch_size, -1)).to(device)
			current_Q1, current_Q2 = self.critic(state, self.actor(state, context1))
			#  = self.op_critic(state, self.actor(state, context1))
			current_QQ1, current_QQ2 = self.critic(state, self.actor(state, context2))
			#  = self.op_critic(state, self.actor(state, context2))
			# current_Q3, _ = self.critic(state, self.actor(state, context3))
			# _ , current_Q4 = self.critic(state, self.actor(state, context4))
			# current_Q3, _ = self.single_critic(state, self.actor(state, context3))
			# _, current_Q4 = self.single_critic(state, self.actor(state, context4))
			actor_loss = -torch.min(current_Q1, current_Q2).mean() - torch.max(current_QQ1, current_QQ2).mean()
			# actor_loss = - 0.5 * (current_Q1.mean() + current_QQ2.mean())
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# for param, target_param in zip(self.op_critic.parameters(), self.op_target.parameters()):
			#	target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.disc.parameters(), self.disc_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
