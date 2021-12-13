from typing import NewType, Dict, List, Callable, cast
from labml import monit, tracker, logger, experiment
from labml.configs import BaseConfigs, option
from itertools import permutations
from labml import experiment
import numpy as np
# from labml_nn.cfr.infoset_save import InfoSetSaver
# from labml_nn.cfr import History as _History, InfoSet as _InfoSet, Action, Player, CFRConfigs

Player = NewType('Player', int)
Action = NewType('Action', str)

ACTIONS = cast(List[Action], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
CHANCES = cast(List[Action], [])
# 0 1 2		9 is a lost turn if a move is made on a taken location
# 3 4 5
# 6 7 8
WINNING_COMBOS = [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8'],
				  ['0', '3', '6'], ['1', '4', '7'], ['2', '5', '8'],
				  ['0', '4', '8'], ['2', '4', '6']]

# Spaghetti code to get the set of winning combos in the way I want
WIN_CONDITION = []

for wc in WINNING_COMBOS:
	WIN_CONDITION.append(list(permutations(wc)))

WIN_CON_SET = [] # <---- This is the variable that contains all lists of winning combinations

for wc in WIN_CONDITION:
	for tup in wc:
		WIN_CON_SET.append(tup)

DRAW_CON = ('0', '1', '2', '3', '4', '5', '6', '7', '8')

PLAYERS = cast(List[Player], [1, 2])

# Class for Histories of actions by player(s)
class History:

	history: str
	winner: float

	def __init__(self, history : str = ''):
		self.history = history
		self.winner = 0

	def is_terminal(self):
		p1_moves = [self.history[index] for index in range(0, len(self.history), 2)]
		p2_moves = [self.history[index] for index in range(1, len(self.history), 2)]
		p1_combs = list(permutations(p1_moves, 3))
		p2_combs = list(permutations(p2_moves, 3))

		# Impossible for tictactoe to terminate with less than 5 moves taken
		if len(self.history) < 5:
			# self.game_over = False
			return False

		# P1 has a winning combination
		for comb in p1_combs:
			if comb in WIN_CON_SET:
				# print("P1 WINS")
				# print(comb)
				self.winner = 1
				# self.game_over = True
				return True
		
		# P2 has a winning combination		
		for comb in p2_combs:
			if comb in WIN_CON_SET:
				# print("P2 WINS")
				# print(comb)
				self.winner = 2
				# self.game_over= True
				return True

		# All spaces are taken, resulting in a tie
		moves = []
		for e in p1_moves:
			if e != '9':
				moves.append(e)
		for e in p2_moves:
			if e != '9':
				moves.append(e)
		if DRAW_CON in list(permutations(moves, 9)):
			# print("DRAW")
			return True
			
	def _terminal_utility_p1(self) -> float:
		util = 0
		if self.is_terminal():
			if self.winner == 1:
				util += 1
			elif self.winner == 2: 
				util -= 1
			elif self.winner == 0:
				return util
		return util


	def terminal_utility(self, i: Player) -> float:
		# This is a zero sum game, so utils are (-1, 1), (1, -1), or (0, 0)
		if i == PLAYERS[0]:
			return self._terminal_utility_p1()
		else:
			return -1 * self._terminal_utility_p1()		
		# raise NotImplementedError()

	def is_chance(self) -> bool:
		raise NotImplementedError()

	def sample_chance(self) -> Action:
		raise NotImplementedError()

	def __add__(self, other: Action):
		return History(self.history + other)

	def player(self) -> Player:
		return cast(Player, (len(self.history) % 2) + 1)

	# Info set key for current history. String of actions only visible to current player
	def info_set_key(self) -> str:
		i = self.player()
		return self.history[i] + self.history[2:]

	def new_info_set(self) -> 'InfoSet':
		return InfoSet(self.info_set_key())

	def __repr__(self):
		return repr(self.history)

def create_new_history():
	return History()



# Class for information sets
class InfoSet:
	key: str
	strategy: Dict[Action, float]
	regret: Dict[Action, float]
	cumulative_strategy: Dict[Action, float]

	def __init__(self, key: str):
		self.key = key
		self.regret = {a: 0 for a in self.actions()}
		self.cumulative_strategy = {a: 0 for a in self.actions()}
		self.calculate_strategy()

	def actions(self) -> List[Action]:
		return ACTIONS

	# Load information set from a saved dictionary
	@staticmethod
	def from_dict(data: Dict[str, any]) -> 'InfoSet':
		pass

	# Calculate current strategy using regret matching
	def calculate_strategy(self):
		regret = {a: max(r, 0) for a, r in self.regret.items()}
		regret_sum = sum(regret.values())
		if regret_sum > 0:
			self.strategy = {a: r / regret_sum for a, r in regret.items()}
		else:
			count = len(list(a for a in self.regret))
			self.strategy = {a: 1 / count for a, r in regret.items()}

	# Get average strategy
	def get_average_strategy(self):
		cum_strategy = {a: self.cumulative_strategy.get(a, 0.) for a in self.actions()}
		strategy_sum = sum(cum_strategy.values())
		if strategy_sum > 0:
			return {a: s / strategy_sum for a, s in cum_strategy.items()}
		else:
			count = len(list(a for a in cum_strategy))
			return {a: 1 / count for a, r in cum_strategy.items()}

	def __repr__(self):
		raise NotImplementedError()

# Class for counterfactual regret minimization algorithm
class CFR:
	info_sets: Dict[str, InfoSet]

	def __init__(self, *,
				create_new_history: Callable[[], History],
				epochs: int,
				n_players: int = 2):
		self.n_players = n_players
		self.epochs = epochs
		self.create_new_history = create_new_history

		self.info_sets = {}

		self.tracker = InfoSetTracker()

	def _get_info_set(self, h: History):
		info_set_key = h.info_set_key()
		if info_set_key not in self.info_sets:
			self.info_sets[info_set_key] = h.new_info_set()
		return self.info_sets[info_set_key]

	# Walks the game tree
	def walk_tree(self, h: History, i: Player, pi_i: float, pi_neg_i: float) -> float:
		if h.is_terminal():
			return h.terminal_utility(i)
		elif h.is_chance():
			a = h.sample_chance()
			return self.walk_tree(h + a, pi_i, pi_neg_i)
		
		I = self._get_info_set(h)
		v = 0
		va = {}
		for a in I.actions():
			if i == h.player():
				va[a] = self.walk_tree(h + a, pi_i * I.strategy[a], pi_neg_i)
			else: 
				va[a] = self.walk_tree(h + a, pi_i, pi_neg_i * I.strategy[a])
			v = v + I.strategy[a] * va[a]

		if h.player() == i:
			for a in I.actions():
				I.cumulative_strategy[a] = I.cumulative_strategy[a] + pi_i * I.strategy[a]
			for a in I.actions():
				I.regret[a] += pi_neg_i * (va[a] - v)
			I.calculate_strategy()

		return v

	def iterate(self):
		for t in monit.iterate('Train', self.epochs):
			for i in range(self.n_players):
				self.walk_tree(self.create_new_history(), cast(Player, i), 1, 1)

			tracker.add_global_step()
			self.tracker(self.info_sets)
			tracker.save()

			if (t + 1) % 1_000 == 0:
				experiment.save_checkpoint()

		logger.inspect(self.info_sets)

# class InfoSetTracker:
# 	def __init__(self):
# 		tracker.set_histogram(f'strategy.*')
# 		tracker.set_historgram(f'average_strategy.*')
# 		tracker.set_histogram(f'regret.*')

# 	def __call__(self, info_sets: Dict[str, InfoSet]):
# 		for I in info_sets.values():
# 			avg_strategy = I.get_average_strategy()
# 			for a in I.actions():
# 				tracker.add({
# 					f'strategy.{I.key}.{a}': I.strategy[a],
# 					f'average_strategy.{I.key}.{a}': avg_strategy[a],
# 					f'regret.{I.key}.{a}': I.regret[a],
# 				})

# class CFRConfigs(BaseConfigs):
# 	create_new_history: Callable[[], History]
# 	epochs: int = 1_00_000
# 	cfr: CFR = 'simple_cfr'

# @option(CFRConfigs.cfr)
# def simple_cfr(c: CFRConfigs):
# 	return CFR(create_new_history = c.create_new_history,
# 			   epochs = c.epochs)

# class Configs(CFRConfigs):
# 	pass

# @option(Configs.create_new_history)
# def _cnh():
# 	return create_new_history

def run_lttt():
	hist = create_new_history()
	taken_spaces = []
	board = ['-', '-', '-', '-', '-', '-', '-', '-', '-']

	while not hist.is_terminal():
		# If player 1 is moving (X)
		if len(hist.history) % 2 == 0:
			# TODO: Fix logic on how to decide what move to make
			move = np.random.randint(9)
			if move != 9 and move not in taken_spaces:
				board[move] = 'X'
				taken_spaces.append(move)
				hist.history = f'{hist.history}{move}'
			elif move in taken_spaces:
				hist.history = f'{hist.history}9'
		# If player 2 is moving (O)
		else:
			# TODO: Fix logic on how to decide what move to make
			move = np.random.randint(9)
			if move != 9 and move not in taken_spaces:
				board[move] = 'O'
				taken_spaces.append(move)
				hist.history = f'{hist.history}{move}'
			elif move in taken_spaces:
				hist.history = f'{hist.history}9'
		print(hist.history)
	draw_board(board)
	return hist

def draw_board(board):
	for i in range(len(board)):
		if i == 0 or i == 3 or i == 6:
			print(f'{board[i]}\t|\t{board[i+1]}\t|\t{board[i+2]}\n')
		
	# for i in board:
	# 	print(i)

# def display_board(board):
# 	for int i 


if __name__ == "__main__":
	test_hist = run_lttt()
	utilities = (test_hist.terminal_utility(1), test_hist.terminal_utility(2))
	print(utilities)
	