import numpy as np


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.num_rounds = num_rounds
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_users = num_users
        self.arms_thresh = arms_thresh
        # arms_thresh - minimal watches per show for it to not get cancelled
        self.users_distribution = users_distribution

        self.last_chosen_arm = None
        self.last_user = None
        self.rewards = np.zeros((num_users, num_arms))
        # num of times each type of user was in our system and used specific arm
        self.counts = np.zeros((num_users, num_arms))

        self.inactive_arms = set()  # set of arms that left the system

        self.t = 0
        self.phase_num = 0
        self.used = np.zeros(num_arms)
        self.uses_left = np.array(arms_thresh, copy=True)

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """

        self.last_user = user_context

        if self.phase_num <= self.num_arms + 4:
            # Explore - each arm has to pass its threshold number on this stage
            has_to_be_used = [i for i in range(self.num_arms) if self.used[i] < self.arms_thresh[i]]
            if len(has_to_be_used) == 0:
                self.last_chosen_arm = np.argmin(self.counts[user_context])
            else:
                # we want to use arm that is least used by user!!!!! implement it
                least_used = float("inf")
                for i in has_to_be_used:
                    if least_used > self.counts[user_context][i]:
                        self.last_chosen_arm = i
                        least_used = self.counts[user_context][i]
        else:
            # Exploit: Choose the arm with the highest estimated value
            arm = np.argmax(self.rewards[user_context])
            loops_left = self.phase_len - int(np.sum(self.arms_thresh) / self.num_arms)
            if self.t >= self.phase_len - int(np.sum(self.arms_thresh) / self.num_arms):
                has_to_be_used = [i for i in range(self.num_arms) if
                                  self.used[i] < self.arms_thresh[i] <= self.used[i] + loops_left]
                if arm not in has_to_be_used and len(has_to_be_used) != 0:
                    max_reward = float("-inf")
                    for i in has_to_be_used:
                        if max_reward < self.rewards[user_context][i] and i not in self.inactive_arms:
                            arm = i
                            max_reward = self.rewards[user_context][i]
            if arm not in self.inactive_arms:
                self.last_chosen_arm = arm
        return self.last_chosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        self.t += 1
        self.used[self.last_chosen_arm] += 1
        self.uses_left[self.last_chosen_arm] -= 1
        # print("arm used: " + str(self.last_chosen_arm))
        # print("uses left: " + str(self.uses_left[self.last_chosen_arm]))

        if self.t == self.phase_len:
            for arm in range(self.num_arms):
                if self.uses_left[arm] > 0:
                    self.inactive_arms.add(arm)
            self.t = 0
            self.phase_num += 1
            self.used = np.zeros(self.num_arms)
            self.uses_left = np.array(self.arms_thresh, copy=True)
        # meaning we're on the first round of exploit
        if self.phase_num == self.num_arms + 5 and self.num_arms > self.num_users:
            num_arms_to_del = self.num_arms - self.num_users
            to_del = np.sum(self.rewards, axis=0)
            for arm in range(self.num_arms):
                if self.arms_thresh[arm] > 0:
                    sorted_idx = np.argpartition(to_del, num_arms_to_del)
                    for i in range(num_arms_to_del):
                        if len(self.inactive_arms) < num_arms_to_del:
                            self.inactive_arms.add(sorted_idx[i])
                else:
                    to_del[arm] = float('inf')
                    sorted_idx = np.argpartition(to_del, num_arms_to_del)
                    for i in range(num_arms_to_del):
                        if len(self.inactive_arms) < num_arms_to_del:
                            self.inactive_arms.add(sorted_idx[i])

        self.counts[self.last_user, self.last_chosen_arm] += 1
        n = self.counts[self.last_user][self.last_chosen_arm]
        value = self.rewards[self.last_user][self.last_chosen_arm]
        self.rewards[self.last_user][self.last_chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id"
