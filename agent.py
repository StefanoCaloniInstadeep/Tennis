import torch
import torch.nn.functional as F
from network import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64*2*2
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 1
WAIT_SIZE = BATCH_SIZE  # int(1e3)

FC = 64


class Agent:
    def __init__(self, state_size, action_size, device=torch.device("cpu")):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.actor_local = ActorNetwork(state_size, action_size).to(device)
        self.actor_target = ActorNetwork(state_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=LR)

        # initialize critic
        self.critic_local = CriticNetwork(state_size, action_size).to(device)
        self.critic_target = CriticNetwork(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
        self.t_step = 0

    def __repr__(self):
        return "Agent\nActor: {}\nCritic: {}".format(self.actor_local.__repr__(), self.critic_local.__repr__())

    def save(self, path):
        torch.save(self.actor_local.state_dict(), path+"_actor_local")
        torch.save(self.actor_target.state_dict(), path+"_actor_target")
        torch.save(self.critic_local.state_dict(), path+"_critic_local")
        torch.save(self.critic_target.state_dict(), path+"_critic_target")

    def load(self, path):
        self.actor_local.load_state_dict(torch.load(path+"_actor_local"))
        self.actor_target.load_state_dict(torch.load(path+"_actor_target"))
        self.critic_local.load_state_dict(torch.load(path+"_critic_local"))
        self.critic_target.load_state_dict(torch.load(path+"_critic_target"))

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(
                state).float().unsqueeze(0).to(self.device)
            action_values = self.actor_local(state)
        return action_values

    def step(self, state, action, reward, next_state, done):
        self.t_step += 1
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # optimize critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            Q_target_next = self.critic_target(
                next_states, target_actions)
            Q_targets = rewards + (GAMMA * Q_target_next * (1-dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # optimize actor
        expected_actions = self.actor_local(states)
        q_expected = self.critic_local(states, expected_actions)
        actor_loss = -torch.mean(q_expected)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update target
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
