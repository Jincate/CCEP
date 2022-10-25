import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, context_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.context = np.zeros((max_size, context_dim))
		self.initial_context = np.zeros((max_size, context_dim))
		self.next_context = np.zeros((max_size, context_dim))
		self.pseudo_reward = np.zeros((max_size, 1))
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, initial_context, context, next_state, next_context, reward, pseudo_reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.context[self.ptr] = context
		self.next_context[self.ptr] = next_context
		self.pseudo_reward[self.ptr] = pseudo_reward
		self.initial_context[self.ptr] = initial_context
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.initial_context[ind]).to(self.device),
			torch.FloatTensor(self.context[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.next_context[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.pseudo_reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)