import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class SeperateActor(nn.Module):
	def __init__(self, state_dim, num_skills, action_dim, max_action):
		super(DoubleActor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.l4 = nn.Linear(state_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, action_dim)
		
		self.l7 = nn.Linear(state_dim, 256)
		self.l8 = nn.Linear(256, 256)
		self.l9 = nn.Linear(256, action_dim)

		self.l10 = nn.Linear(state_dim, 256)
		self.l11 = nn.Linear(256, 256)
		self.l12 = nn.Linear(256, action_dim)
		self.max_action = max_action
		

	def forward(self, state, context):
		# su = torch.cat([state, context], dim=1)
		a1 = F.relu(self.l1(state))
		a1 = F.relu(self.l2(a1))
		a1 = self.max_action * torch.tanh(self.l3(a1))
		
		a2 = F.relu(self.l4(state))
		a2 = F.relu(self.l5(a2))
		a2 = self.max_action * torch.tanh(self.l6(a2))
		
		a3 = F.relu(self.l7(state))
		a3 = F.relu(self.l8(a3))
		a3 = self.max_action * torch.tanh(self.l9(a3))

		a4 = F.relu(self.l10(state))
		a4 = F.relu(self.l11(a4))
		a4 = self.max_action * torch.tanh(self.l12(a4))
		
		a = torch.cat((a1, a2, a3, a4), 0)
		index = torch.nonzero(context)
		index = (index[:,0] + index[:,1] * context.shape[0]).reshape(-1)
		return torch.index_select(a, dim = 0, index = index)

class Centralized_Actor(nn.Module):
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

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, -q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class CCEP(object):
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

		self.actor = Centralized_Actor(state_dim, num_skills, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = DoubleCritic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.num_skills = num_skills
		
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

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state,  action, context, next_state, next_context, reward, not_done = replay_buffer.sample(batch_size)
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			next_action = (
				self.actor_target(next_state, next_context) + noise
			).clamp(-self.max_action, self.max_action)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q1 = reward + not_done * self.discount * target_Q1
			target_Q2 = reward + not_done * self.discount * target_Q2
			target_Q = torch.min(target_Q1, target_Q2)
			
		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			context1 = np.zeros((batch_size, self.num_skills))
			context1[:, 0] = 1
			context1 = torch.FloatTensor(context1.reshape(batch_size, -1)).to(device)
			context2 = np.zeros((batch_size, self.num_skills))
			context2[:, 1] = 1
			context2 = torch.FloatTensor(context2.reshape(batch_size, -1)).to(device)
			context3 = np.zeros((batch_size, self.num_skills))
			context3[:, 2] = 1
			context3 = torch.FloatTensor(context3.reshape(batch_size, -1)).to(device)
			context4 = np.zeros((batch_size, self.num_skills))
			context4[:, 3] = 1
			context4 = torch.FloatTensor(context4.reshape(batch_size, -1)).to(device)
			action1 = self.actor(state, context1)
			action2 = self.actor(state, context2)
			action3 = self.actor(state, context3)
			action4 = self.actor(state, context4)
			current_Q1, current_Q2 = self.critic(state, action1)
			current_QQ1, current_QQ2 = self.critic(state, action2)
			current_Q3, _ = self.critic(state, action3)
			_ , current_Q4 = self.critic(state, action4)
			actor_loss = 0.25 * (-torch.min(current_Q1, current_Q2).mean() - torch.max(current_QQ1, current_QQ2).mean() - current_Q3.mean() - current_Q4.mean())
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			loss = F.mse_loss(action1, action2)
			self.logger.add_scalar("loss", loss, self.total_it)
			
			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
		
